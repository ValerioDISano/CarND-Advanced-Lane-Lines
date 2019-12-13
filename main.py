#!/usr/bin/env python3

from utilities import ImagesLoader
from utilities import VideoLoader
from utilities import Visualizer
from Line import ImageProcessor
import argparse
import sys

def main():

    ap = argparse.ArgumentParser(description="Lane detection")
    ap.add_argument("-d", "--directory", required=True, type=str, help="Data files directory")
    ap.add_argument("-t", "--file_type", required=True, type=str, help="Choose between videos or images as input")

    args = vars(ap.parse_args())
    test_path = args["directory"]
    file_type = args["file_type"]

    print("Selected directory: {}".format(test_path))
    print("Selected filte type: {}".format(file_type))

    if file_type != "image" and file_type != "video":
        sys.stderr.write("Error!!! Unrecognized option -t/--file_type option.")
        print(ap.print_help())
    
    ### PARAMETERS ###
    
    ### Camera Calibration ###
    #calibration_imgs_path = "./camera_cal"
    #calibration_grid_size = (9,6)
    ### Thresholding ###
    filter_data = lambda path : path.endswith('project_video.mp4')

    loader = ImagesLoader(test_path)\
            if file_type == "image"\
            else VideoLoader(test_path, predicate=filter_data)
    
    
    processor = ImageProcessor()
    frame_processor = lambda frame : processor[frame]
    
    loader.processData(frame_processor)
    loader.writeOutput("video_out.mp4")


    #for img in loader.dataIterable():

        #Visualizer.show(img)

        #out = processor[img]     
        #Visualizer.show(out)
        #Visualizer.showContinuosly(out)
        #Visualizer.write

    return



if __name__ == "__main__":
    main()
