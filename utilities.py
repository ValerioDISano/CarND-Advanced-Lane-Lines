import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import pygame
import os
import sys
import numpy as np
import cv2

"""
    Set of utilities function
"""


class DataLoader(object):
    """
    Class to load data
    """

    def __init__(self, folder_path, predicate=None):
        # folder_path : folder containing data
        # predicate : function to filter data 
        if predicate is None:
            self.file_list = [os.path.join(folder_path,x) for x in os.listdir(folder_path)]
        else:
            self.file_list = [os.path.join(folder_path,x) for x in os.listdir(folder_path) if predicate(x)]

        self.loaded_data = None
        self.current_idx = int(-1)
        self.n_file = len(self.file_list)

    def getFileList(self):
        return file_list

class ImagesLoader(DataLoader):
    """
    Class to load and process images 
    """
    def __init__(self, folder_path, predicate=None):
        super(ImagesLoader, self).__init__(folder_path, predicate)
        self.data_out = {} 

    def getNextImg(self):
        self.current_idx += 1
        self.current_idx %= self.n_file

        self.loaded_data = mpimg.imread(\
                            self.file_list[self.current_idx]\
                            )
        return self.loaded_data 
    def getImg(self, idx):
         return mpimg.imread(\
                  self.file_list[idx]\
                  )

    def showImg(self, idx):
        
        try:
            plt.figure()
            plt.imshow(self.getImg(idx))
            plt.show()
        except:
            print("Index out of boundaries")
            sys.exit(1)

    def showAll(self):
        print("Number of images to print: {}".format(self.n_file))
        [self.showImg(x) for x in range(self.n_file)]
        return
        
    def dataIterable(self):
        # function to get an iterable for the images
        for _ in range(self.n_file):
            yield self.getNextImg()
    
    def processData(self, function):
        # Process images with function
        for idx, img in enumerate(self.dataIterable()):
            self.data_out[os.path.basename(self.file_list[idx])] = function(img)

    def writeOutput(self, path):
        # save processed images
        for name, img in self.data_out.items():
            mpimg.imsave(os.path.join(path,name), img)

class VideoLoader(DataLoader):
    """
    Class to load videos
    """
    def __init__(self, folder_path, predicate=None):
        super(VideoLoader, self).__init__(folder_path, predicate)

    def getNextVideo(self):

        self.current_idx += 1
        self.current_idx %= self.n_file

        self.loaded_data = VideoFileClip(\
                self.file_list[self.current_idx]\
                )

        return self.loaded_data

    def getVideo(self, idx):

        return VideoFileClip(\
                self.file_list[idx]\
                )

    def showVideo(self, idx):

        try:
            clip = self.getVideo(idx)
            clip.preview()
        except:
            print("Index out of boundaries")
            sys.exit(1)

    def dataIterable(self):

        self.getNextVideo()

        for frame in self.loaded_data.iter_frames():
            yield frame

    def processData(self, function):
        
        self.getNextVideo()

        self.data_out = self.loaded_data.fl_image(function)

    def writeOutput(self, path):

        if self.data_out is not None:
            self.data_out.write_videofile(path, audio=False)

class Visualizer(object):

    def __init__(self, img):
        self.img = img
        plt.ion()
    
    @staticmethod
    def polygonOverlap(img, vertices, color=(255,0,0)):
        
        pts = vertices.reshape((-1,1,2)).astype(np.int32)
        poly_img = np.copy(img)
        cv2.polylines(poly_img,[pts], False, color)
        
        dst = cv2.addWeighted(img, 0.5, poly_img, 0.5, 0.0)

        return Visualizer.show(dst)
    
    @staticmethod
    def show(img):
        plt.figure()
        plt.imshow(img)
        plt.show()
    
    @staticmethod
    def showContinuosly(img, frame_rate=60):
        
        plt.imshow(img)
        plt.draw()
        plt.pause(1.0/frame_rate)

        return True

def incrementalAverage(avg, x, n):

    return (avg + ((x-avg)/n))


def getNonZero(img):
    nonzero_pixels = img.nonzero()
    return nonzero_pixels[0], nonzero_pixels[1]

