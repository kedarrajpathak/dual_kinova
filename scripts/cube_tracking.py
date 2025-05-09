import cv2
import numpy as np
import math
import os
import time
from scipy.spatial.transform import Rotation as R

class KalmanFilter3D:
    """Simple Kalman filter for tracking 3D position and orientation"""
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        # State: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
        self.kalman = cv2.KalmanFilter(10, 7)  # 10 state variables, 7 measurements (x,y,z,qw,qx,qy,qz)
        
        # Transition matrix (state update matrix)
        self.kalman.transitionMatrix = np.eye(10, dtype=np.float32)
        # Position + velocity model
        self.kalman.transitionMatrix[0, 3] = 1.0  # x += vx
        self.kalman.transitionMatrix[1, 4] = 1.0  # y += vy
        self.kalman.transitionMatrix[2, 5] = 1.0  # z += vz
        
        # Measurement matrix (maps state to measurement)
        self.kalman.measurementMatrix = np.zeros((7, 10), dtype=np.float32)
        self.kalman.measurementMatrix[0, 0] = 1.0  # x
        self.kalman.measurementMatrix[1, 1] = 1.0  # y
        self.kalman.measurementMatrix[2, 2] = 1.0  # z
        self.kalman.measurementMatrix[3, 6] = 1.0  # qw
        self.kalman.measurementMatrix[4, 7] = 1.0  # qx
        self.kalman.measurementMatrix[5, 8] = 1.0  # qy
        self.kalman.measurementMatrix[6, 9] = 1.0  # qz
        
        # Process noise
        self.kalman.processNoiseCov = np.eye(10, dtype=np.float32) * process_noise
        # Higher process noise for velocities
        self.kalman.processNoiseCov[3:6, 3:6] *= 10
        
        # Measurement noise
        self.kalman.measurementNoiseCov = np.eye(7, dtype=np.float32) * measurement_noise
        
        # Initial state covariance
        self.kalman.errorCovPost = np.eye(10, dtype=np.float32) * 1.0
        
        self.initialized = False
        
    def update(self, position, orientation):
        """Update the filter with a new measurement"""
        measurement = np.array([
            position[0], position[1], position[2],
            orientation[0], orientation[1], orientation[2], orientation[3]
        ], dtype=np.float32).reshape(-1, 1)
        
        if not self.initialized:
            # Initialize state
            self.kalman.statePost[0:3, 0] = position
            self.kalman.statePost[3:6, 0] = 0  # Initial velocity is zero
            self.kalman.statePost[6:10, 0] = orientation
            self.initialized = True
            return position, orientation
            
        # Predict
        predicted = self.kalman.predict()
        
        # Update with measurement
        corrected = self.kalman.correct(measurement)
        
        # Extract position and orientation
        filtered_pos = corrected[0:3, 0]
        filtered_quat = corrected[6:10, 0]
        
        # Normalize quaternion
        quat_norm = np.linalg.norm(filtered_quat)
        if quat_norm > 0:
            filtered_quat = filtered_quat / quat_norm
            
        return filtered_pos, filtered_quat


class ArucoCubeTracking:
    def __init__(self, 
                 camera_id=0, 
                 aruco_dict_type=cv2.aruco.DICT_4X4_50,
                 marker_size=0.04,  # 40mm
                 cube_size=0.06,    # 60mm
                 frame_rate=30.0,   # Hz
                 calibration_matrix_path='calibration_matrix.npy',
                 distortion_coefficients_path='distortion_coefficients.npy',
                 flip_horizontal=False,
                 flip_vertical=False):
        
        # Store parameters
        self.camera_id = camera_id
        self.aruco_dict_type = aruco_dict_type
        self.marker_size = marker_size
        self.cube_size = cube_size
        self.frame_rate = frame_rate
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        
        # Load camera calibration
        try:
            print(f"Loading calibration from {calibration_matrix_path}")
            self.camera_matrix = np.load(calibration_matrix_path)
            self.distortion_coefficients = np.load(distortion_coefficients_path)
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            # Use a default calibration (will be less accurate)
            self.camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
            self.distortion_coefficients = np.zeros(5, dtype=np.float32)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.image_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Setup marker-to-cube transforms
        self.setup_marker_to_cube_transforms()
        
        # Initialize Kalman filters for tracking
        self.left_hand_filter = KalmanFilter3D()
        self.right_hand_filter = KalmanFilter3D()
        
        # Store latest poses
        self.left_hand_pose = None  # (position, quaternion)
        self.right_hand_pose = None  # (position, quaternion)
        self.left_hand_markers = []
        self.right_hand_markers = []
        
        print("ArUco cube tracking initialized")
    
    def setup_marker_to_cube_transforms(self):
        """
        Setup transforms from each marker to the cube origin.
        For a cube of side 1, the origin is at (0.5, 0.5, 0.5)
        """
        cube_side = self.cube_size  # Cube size in meters
        half_side = cube_side / 2.0
        
        # Define transforms from each marker to cube origin
        # Each tuple is (translation, rotation_matrix)
        self.marker_to_cube = {
            0: (np.array([0, 0, -half_side]), 
                R.from_euler('xyz', [0, 0, 0]).as_quat()),
            
            1: (np.array([0, 0, -half_side]), 
                R.from_euler('xyz', [-math.pi/2, 0, 0]).as_quat()),
            
            2: (np.array([0, 0, -half_side]), 
                R.from_euler('xyz', [math.pi, 0, 0]).as_quat()),
            
            3: (np.array([0, 0, -half_side]), 
                R.from_euler('xyz', [math.pi/2, 0, 0]).as_quat()),
            
            4: (np.array([0, 0, -half_side]), 
                R.from_euler('yxz', [math.pi/2, math.pi, 0]).as_quat()),
            
            5: (np.array([0, 0, -half_side]), 
                R.from_euler('yxz', [-math.pi/2, math.pi, 0]).as_quat()),
        }
        
        for i in range(6):
            cube_translation, cube_quaternion = self.marker_to_cube[i]
            
            cube_rotation_matrix = R.from_quat(cube_quaternion).as_matrix()
            cube_transform = np.eye(4)
            cube_transform[:3, :3] = cube_rotation_matrix
            cube_transform[:3, 3] = cube_translation
            self.marker_to_cube[i] = cube_transform
            
        # Right hand cube uses the same transforms but with different IDs (6-11)
        for i in range(6):
            self.marker_to_cube[i + 6] = self.marker_to_cube[i]
    
    def process_frame(self):
        """Process a frame from the camera"""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            return None
        
        # Apply flipping if required
        if self.flip_horizontal and self.flip_vertical:
            frame = cv2.flip(frame, -1)  # Both horizontally and vertically
        elif self.flip_horizontal:
            frame = cv2.flip(frame, 1)  # Horizontal
        elif self.flip_vertical:
            frame = cv2.flip(frame, 0)  # Vertical
        
        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        output_frame = frame.copy()
        
        if ids is not None and len(ids) > 0:
            # Group markers by left/right hand
            self.left_hand_markers = []
            self.right_hand_markers = []
            
            # Draw markers and estimate poses
            cv2.aruco.drawDetectedMarkers(output_frame, corners, ids)
            
            for i, marker_id in enumerate(ids.flatten()):
                # Estimate pose
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], self.marker_size, self.camera_matrix, self.distortion_coefficients
                )
                
                # Draw axis for each marker
                cv2.drawFrameAxes(output_frame, self.camera_matrix, self.distortion_coefficients, 
                                  rvec, tvec, self.marker_size)
                
                # Convert rotation vector to rotation matrix
                rot_matrix, _ = cv2.Rodrigues(rvec[0][0])
                
                # Get marker to cube transform
                if marker_id in self.marker_to_cube:                    
                    # Convert marker pose to cube pose
                    marker_transform = np.eye(4)
                    marker_transform[:3, :3] = rot_matrix
                    marker_transform[:3, 3] = tvec[0][0]
                    
                    cube_transform = self.marker_to_cube[marker_id]
                    
                    # Calculate cube pose in camera frame
                    cube_pose = np.dot(marker_transform, cube_transform)
                    cube_pos = cube_pose[:3, 3]
                    cube_rot_matrix = cube_pose[:3, :3]
                    cube_quat = R.from_matrix(cube_rot_matrix).as_quat()
                    
                    # Group by marker ID
                    if 0 <= marker_id <= 5:  # Left hand
                        self.left_hand_markers.append((marker_id, cube_pos, cube_quat))
                    elif 6 <= marker_id <= 11:  # Right hand
                        self.right_hand_markers.append((marker_id, cube_pos, cube_quat))
    
            output_frame = cv2.flip(output_frame, 1)  # Flip the output frame horizontally
            
            # Update left hand cube position if any markers were detected
            if self.left_hand_markers:
                # Average positions from all markers
                avg_position = np.mean([marker[1] for marker in self.left_hand_markers], axis=0)
                
                # Average orientations (quaternions)
                avg_quaternion = np.mean([marker[2] for marker in self.left_hand_markers], axis=0)
                avg_quaternion = avg_quaternion / np.linalg.norm(avg_quaternion)  # Normalize
                
                # Update filter with averaged pose
                left_hand_pos, left_hand_quat = self.left_hand_filter.update(
                    avg_position, avg_quaternion
                )
                # Store the pose
                self.left_hand_pose = (left_hand_pos, left_hand_quat)
                
                # Display position in the image
                cv2.putText(output_frame, f"Left: {left_hand_pos[0]:.2f}, {left_hand_pos[1]:.2f}, {left_hand_pos[2]:.2f}", 
                           (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Update right hand cube position if any markers were detected
            if self.right_hand_markers:
                # Average positions from all markers
                avg_position = np.mean([marker[1] for marker in self.right_hand_markers], axis=0)
                
                # Average orientations (quaternions)
                avg_quaternion = np.mean([marker[2] for marker in self.right_hand_markers], axis=0)
                avg_quaternion = avg_quaternion / np.linalg.norm(avg_quaternion)  # Normalize
                
                # Update filter with averaged pose
                right_hand_pos, right_hand_quat = self.right_hand_filter.update(
                    avg_position, avg_quaternion
                )
                # Store the pose
                self.right_hand_pose = (right_hand_pos, right_hand_quat)
                
                # Display position in the image
                cv2.putText(output_frame, f"Right: {right_hand_pos[0]:.2f}, {right_hand_pos[1]:.2f}, {right_hand_pos[2]:.2f}", 
                           (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return output_frame, self.left_hand_pose, self.right_hand_pose
    
    def get_left_hand_pose(self):
        """Return the latest left hand cube pose"""
        return self.left_hand_pose
    
    def get_right_hand_pose(self):
        """Return the latest right hand cube pose"""
        return self.right_hand_pose
    
    def run(self):
        """Run the main processing loop"""
        try:
            while True:
                # start_time = time.time()
                
                # Process frame
                output_frame = self.process_frame()
                if output_frame is None:
                    break
                
                # Display the resulting frame
                cv2.imshow('ArUco Cube Tracking', output_frame)
                
                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # # Control the frame rate
                # elapsed = time.time() - start_time
                # sleep_time = max(0, 1.0/self.frame_rate - elapsed)
                # if sleep_time > 0:
                #     time.sleep(sleep_time)
                
        finally:
            # Release resources when done
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    # Example calibration file paths - update these to your actual paths
    calibration_matrix_path = "calibration_matrix.npy"
    distortion_coefficients_path = "distortion_coefficients.npy"
    
    # Create and run the tracker
    tracker = ArucoCubeTracking(
        camera_id=0,
        aruco_dict_type=cv2.aruco.DICT_4X4_50,
        marker_size=0.04,  # 4cm
        cube_size=0.06,    # 6cm
        calibration_matrix_path=calibration_matrix_path,
        distortion_coefficients_path=distortion_coefficients_path
    )
    
    tracker.run()


if __name__ == '__main__':
    main()