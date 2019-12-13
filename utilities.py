import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import pygame
import os
import sys
import numpy as np
import cv2

class DataLoader(object):

    def __init__(self, folder_path, predicate=None):
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

    def __init__(self, folder_path, predicate=None):
        super(ImagesLoader, self).__init__(folder_path, predicate)

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
        for _ in range(self.n_file):
            yield self.getNextImg()


class VideoLoader(DataLoader):

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
    #print(nonzero_pixels)
    return nonzero_pixels[0], nonzero_pixels[1]

