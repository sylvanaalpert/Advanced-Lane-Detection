import os
import cv2
import numpy as np


## Pipeline:
# Undistort
# Color and gradient thresholds (including magnitude and direction)
# Perspective Transform
# Peaks in histogram and sliding Window
# Search in found locations for subsequent frames
# Performance metrics: similar curvature, distance from center roughly equal to both sides, parallel
# Redo blind search in case performance decreases. 
# Smooth over previous n frames
# Project back to the road and create output video









class CameraCalibration(object):
    
    def __init__(self, image_dir, nx, ny):
        
        self.image_directory = image_dir
        self.nx = nx
        self.ny = ny
        self.camera_mtx = np.zeros((3,3), np.float32)
        self.dist_coeff = np.zeros((1,5), np.float32)
        
        self.calculate()
    
    def calculate(self):
        
        filenames = os.listdir(self.image_directory)
        
        imgpoints = []
        objpoints = []
        objp = np.zeros((self.nx*self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2) # x and y coordinates

        for fn in filenames: 
            
            file = os.path.join(self.image_directory, fn)
            img = cv2.imread(file)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
        
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.camera_mtx = mtx
        self.dist_coeff = dist    
        
    def get_params(self):
        return self.camera_mtx, self.dist_coeff
        

        

            