import numpy as np
import cv2
from PolyFitter import PolyFit

class SlidingWindow(object):

    def __init__(self, n_windows=9, margin=100, min_pix_number=50):

        self.left_fitter = PolyFit(2)
        self.right_fitter = PolyFit(2)

        self.is_initialized = False

        self.img_shape = None
        self.window_height = None
        self.leftx_base = None
        self.rightx_base = None

        self.n_windows = n_windows
        self.margin = margin
        self.min_pix_number = min_pix_number

        self.left_window = None
        self.right_window = None

    def lanesDetection(self, img):
        
        

    def _init(self, img):
        
        self.img_shape = img.shape
        histogram = np.sum(self.img_shape[0]//2:,:), axis=0)
        self.mid_point = histogram.shape[0]//2

        self.leftx_base = np.argmax(histogram[:midpoint])
        self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        self.window_height = np.int(self.img_shape[0]//self.n_windows)
        
        Window(self.leftx_base,\
                self.img_shape[0]-self.window_height//2,\
                self.margin,\
                self.window_height)
        Window(self.rightx_base,\
                self.img_shape[0]-self.window_height//2,\
                self.margin,\
                self.window_height)
        
        self.is_initialized = True

class Window(object):

    """
        (x1,y2)-----(x2,y2)
          |            |
          |            |
          |            |
        (x1,y1)-----(x2,y1)
    """

    def __init__(self, x0, y0, width, height, progress='decreasing'):
        
        self.w = width
        self.h = height
        self.x1 = x0 - width//2
        
        self.sum = (lambda x, y : x-y)\
                        if progress == 'decreasing'\
                        else (lambda x, y : x+y)
        self.opposite_sum =  (lambda x, y : x+y)\
                        if progress == 'decreasing'\
                        else (lambda x, y : x-y)
 
        
        self.y1 = self.opposite_sum(y0, self.height//2)

        self.y2 = self.sum(self.y1, self.height)
        self.x2 = self.x1 + self.width

    def showOnImage(self, img, color=(0,255,255), width=2):

        out_img = np.copy(img)
        cv2.rectangle(out_img, (self.x1, self.y1), (self.x2, self.y2), color, width)

    def positionStep(self, x0, y0):

        self.moveVerticallt()
        self.updateCenter(x0,y0)

    def moveVertically(self):
        
        self.y1 = self.y2
        self.y2 = self.sum(self.y1, self.height)

    def updateCenter(self, x0, y0):
        self.x1 = x0 - width//2
        self.x2 = x0 + width//2
