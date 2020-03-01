#!/usr/bin/env python3

from utilities import ImagesLoader
from utilities import VideoLoader
from utilities import Visualizer
from Line import ImageProcessor
import argparse
import sys

def main():

    # enput arguments to main script to select data source data type (images or videos)
    # and select the source data folder
    ap = argparse.ArgumentParser(description="Lane detection")
    ap.add_argument("-d", "--directory", required=True, type=str, help="Data files directory")
    ap.add_argument("-t", "--file_type", required=True, type=str, help="Choose between videos or images as input")

    args = vars(ap.parse_args())
    test_path = args["directory"]
    file_type = args["file_type"]

    print("Selected directory: {}".format(test_path))
    print("Selected filte type: {}".format(file_type))

    # Test that the input argumets are valid
    if file_type != "image" and file_type != "video":
        sys.stderr.write("Error!!! Unrecognized option -t/--file_type option.")
        print(ap.print_help())
    
    # filter data is used to select only files with a certain extension or name
    filter_data = (lambda path : path.endswith('project_video.mp4'))\
            if file_type == "video"\
            else (lambda path : path.endswith('.jpg'))
    
    # allocate a Images/VideoLoader class to manage the access to the files to be precessed
    loader = ImagesLoader(test_path)\
            if file_type == "image"\
            else VideoLoader(test_path, predicate=filter_data)
    
    
    # Allocate an ImageProcessor class that is responsible to apply the whole
    # lanes detection pipeline to the data
    if file_type == "video": 
        processor = ImageProcessor()
        frame_processor = lambda frame : processor[frame]
    else: # if the input is a series of images taken from different scenarios is important
          # to reset the class in order to not use previous image computation on the
          # current image
        def frame_processor(frame):
            processor = ImageProcessor()
            return processor[frame]
    
    # Apply the lanes detection pipeline
    loader.processData(frame_processor) 
    
    # Save the result
    loader.writeOutput("video_out.mp4")\
            if file_type == "video"\
            else loader.writeOutput("output_images")

    return



if __name__ == "__main__":
    main()
