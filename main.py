#!/usr/bin/env python3

from utilities import ImagesLoader
from CameraCalibration import CameraCalibrator
from Thresholder import Thresholding

import matplotlib.pyplot as plt

def main():


    ### PARAMETERS ###
    test_imgs_path = "./test_images"
    ### Camera Calibration ###
    calibration_imgs_path = "./camera_cal"
    calibration_grid_size = (9,6)
    ### Thresholding ###
    
    test_loader = ImagesLoader(test_imgs_path)

    calibrator = CameraCalibrator(calibration_grid_size, calibration_imgs_path, visualization=False)
    calibrator.calibrate()

    print(calibrator.calibration_data)
    
    thresholder = Thresholding()

    for img in test_loader.dataIterable():
        cal_img = calibrator.applyCalibration(img)
        
        thresholder.applyThresholding(cal_img,'gradient mag',(50,255))
        thresholder.applyThresholding(cal_img,'gradient dir',(0.7,1.3))
        thresholder.applyThresholding(cal_img,'s channel',(180,255))
    
        binary = thresholder.combine()

        plt.figure()
        plt.imshow(binary)
        plt.show()


    return



if __name__ == "__main__":
    main()
