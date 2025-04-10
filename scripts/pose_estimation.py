'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


import numpy as np
import cv2
import sys
import argparse
import time


ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def pose_esitmation(frame, aruco_dict_type, marker_length, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''
    rvec = []
    tvec = []
    idx = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()


    # corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
    #     cameraMatrix=matrix_coefficients,
    #     distCoeff=distortion_coefficients)
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters)

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            r, t, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_length, matrix_coefficients,
                                                                       distortion_coefficients)
            rvec.append(r[0][0])
            tvec.append(t[0][0])
            idx.append(ids[i][0])
            # print(f"rvec: {r}, tvec: {t}")
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners) 

            # Draw Axis
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, r, t, marker_length*2)  
    
    rvec = np.array(rvec)
    tvec = np.array(tvec)
    # if len(rvec) != 0 and len(tvec) != 0: 
    #     print(f"rvec: {rvec}, tvec: {tvec}, idx: {idx}")
    return frame, rvec, tvec, idx

if __name__ == '__main__':

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    # ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    # ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    # ap.add_argument("-l", "--length", type=float, default=0.1, help="Length of the marker's side in meters")
    # args = vars(ap.parse_args())

    
    aruco_dict_type = ARUCO_DICT["DICT_4X4_50"]  # Update with the desired ArUco dictionary
    marker_length = 0.053  # Length of the marker's side in meters
    calibration_matrix_path = "/workspace/isaaclab/scripts/tutorials/03_envs/calibration_matrix.npy"  # Update with the correct path
    distortion_coefficients_path = "/workspace/isaaclab/scripts/tutorials/03_envs/distortion_coefficients.npy"  # Update with the correct path
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output, rvec, tvec, ids = pose_esitmation(frame, aruco_dict_type, marker_length, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()