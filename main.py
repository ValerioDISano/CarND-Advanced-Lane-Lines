#!/usr/bin/env python3

from utilities import ImagesLoader
from utilities import Visualizer
from Line import ImageProcessor


def main():


    ### PARAMETERS ###
    test_imgs_path = "./test_images"
    ### Camera Calibration ###
    calibration_imgs_path = "./camera_cal"
    calibration_grid_size = (9,6)
    ### Thresholding ###
        
    test_loader = ImagesLoader(test_imgs_path)
    
    """
    calibrator = CameraCalibrator(calibration_grid_size, calibration_imgs_path, visualization=False)
    calibrator.calibrate()

    print(calibrator.calibration_data)
    
    plt.show()
    thresholder = Thresholding()
    transformer = PerspectiveTransformer()

    """
    processor = ImageProcessor()
    for img in test_loader.dataIterable():

        Visualizer.show(img)

        out = processor[img]     
        Visualizer.show(out)
        

    return



if __name__ == "__main__":
    main()
