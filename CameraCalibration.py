import cv2
import numpy as np
from utilities import ImagesLoader
import matplotlib.pyplot as plt

class CameraCalibrator(object):

    def __init__(self, grid_size, data_folder_src, visualization=False):

        self.cal_data_src = data_folder_src
        self.grid_size = grid_size
        self.data_loader = ImagesLoader(data_folder_src)
        self.visualization = visualization
        
        # Lists to store img/obj points for all images in the calibration dataset
        self.all_obj_pts = [] # 3D points in real world coordinates
        self.all_img_pts = [] # 2D points on the image (image plane space)
        
        self.obj_pts = self.computeObjPoints(*grid_size) # Object pts are equals for all images
                                                         # since the grid is constant
       
        self.calibration_data = dict()
        self.is_calibrated = False

    def calibrate(self):
        
        for img in self.data_loader.dataIterable(): # Iterate over all images in the Calibration folder
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find Chessboard Corners
            ret, corners = cv2.findChessboardCorners(gray_img, self.grid_size, None)

            if ret == True: # Check if corners were found
                self.all_img_pts.append(corners)
                self.all_obj_pts.append(self.obj_pts)
                
                if self.visualization: # Visualize corners found in each processed image for debugging purposes
                    img = cv2.drawChessboardCorners(img, self.grid_size, corners, ret)
                    plt.imshow(img)
                    plt.show()

        ret, mtx, dist, rvecs, tvecs =  cv2.calibrateCamera(self.all_obj_pts, \
                                                           self.all_img_pts, \
                                                           gray_img.shape[::-1],\
                                                           None, None)
        # dist = distortion coefficient
        # mtx = matrix needed to perform transformation from 3D points to 2D points
        # rvecs, tvecs = rotation and translation vectors to identify the pos of the camera in the world
        
        self.calibration_data = {"mtx" : mtx, "dist" : dist}
        self.is_calibrated = True
        print("Camera calibration performed")

    def computeObjPoints(self, nx, ny):

        obj_pts = np.zeros((nx*ny, 3), np.float32) # Array of 3D coord
        obj_pts[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) # Create a grid of coordinates.
                                                              # The z coordinate is always zero
        return obj_pts

    def setVisualization(boolean):
        self.visualization = boolean

    def applyCalibration(self, frame):
        
        if len(self.calibration_data) > 0:
            calibrated_frame = cv2.undistort(frame,\
                                             self.calibration_data["mtx"],\
                                             self.calibration_data["dist"],\
                                             None,
                                             self.calibration_data["mtx"])
            return calibrated_frame
        else:
            print("Error! no calibration data found...")
 
