import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilities import Visualizer

class PerspectiveTransformer(object):

    def __init__(self, src_pts=None, offset_percent=0.1):
        
        self.offset = None
        self.offset_percent = offset_percent
        self.img_shape = None
        
        
        self.src = np.float32([[588,470],\
                               [330,655],\
                               [1030,645],\
                               [735,470]]) if src_pts is None else src_pts
        self.src = np.float32([[580, 460],[700, 460],[1040, 680],[260, 680]])

        self.is_src_changed = True
        self.M = None

    def applyTransformation(self, img, visualization=True):

        self._setDestPoints(img)
        
        if self.is_src_changed:
            self.M = cv2.getPerspectiveTransform(self.src, self.dst)
            self.is_src_changed = False
        
        if visualization:
            Visualizer.polygonOverlap(img, self.src)

        warped_img = cv2.warpPerspective(img, self.M, img.shape[:2][::-1])

        return warped_img

    def revertTransformation(self, img):

        unwarped_img = cv2.warpPerspective(img, np.linalg.inv(self.M), img.shape[:2][::-1])

        return unwarped_img

    def _setDestPoints(self, img):
        
        if (self.img_shape is None) or (img.shape != self.img_shape):
            self.img_shape = img.shape
            self.offset = (int(self.img_shape[0]*self.offset_percent),int(self.img_shape[1]*self.offset_percent))
            """self.dst = np.float32([[self.offset[0], self.offset[1]],\
                                  [self.img_shape[0]-self.offset[0], self.offset[1]],\
                                  [self.img_shape[0]-self.offset[0], self.img_shape[1]-self.offset[1]],\
                                  [self.offset[0], self.img_shape[1]-self.offset[1]]])
            """

            self.dst = np.float32([[100, 100],\
                                  [self.img_shape[0]-100, 100],\
                                  [self.img_shape[0]-100, self.img_shape[1]-100],\
                                  [100, self.img_shape[1]-100]])
            self.dst = np.float32([[260, 0],[1040, 0],[1040, 720],[260, 720]])
        return

    def setSrcPoints(self, pts):
        self.src = pts
        self.is_src_changed = True
