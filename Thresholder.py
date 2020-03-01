from utilities import Visualizer

import numpy as np
import cv2
import matplotlib.pyplot as plt

class Thresholding(object):
    """
    It allows to combine several thresholding techniques to an image.
    This class is based on the Factory design pattern
    """
    def __init__(self):
        self.thresholded_intensity_imgs = [] # contains all the images thresholded on the intes
        self.thresholded_gradient_imgs = []
        self.thresholded_sobel_imgs = []

        self.thresh_intervals = {}
        
        self.current_img = None
        self.current_gradient = []
        self._image_shape = None
        
        self.sobel_kernel = 15

    def _threshold(self, img, interval):

        t_img = np.zeros_like(img)
        t_img[(img >= interval[0]) & (img <= interval[1])] = 1

        return t_img

    def applyThresholding(self, rgb_img, method, interval):
        
        img_shape = rgb_img.shape[:2]
        if (self._image_shape is not None) and (img_shape != self._image_shape):
            print("ERROR. Several thresholding before a combine have to be applied on the same image!")
            return
        else:
            self._image_shape = img_shape

        thresholding_fcn = self._getThresholdingMethod(method)
        self.thresh_intervals[method] = interval
        if 'channel' in method:
            self.thresholded_intensity_imgs.append(thresholding_fcn(rgb_img))
        elif "sobel" in method:
            self.thresholded_sobel_imgs.append(thresholding_fcn(rgb_img))
        else:
            self.thresholded_gradient_imgs.append(thresholding_fcn(rgb_img))

    def _getThresholdingMethod(self, method):

        if method == 'gradient mag':
            return self._gradient_mag_thresholding
        elif method == 'gradient dir':
            return self._gradient_dir_thresholding
        elif method == 'sobel x':
            return lambda img : self._Sobel_thresholding(img, 'x')
        elif method == 'sobel y':
            return lambda img : self._Sobel_thresholding(img, 'y') 
        elif method == 's channel':
            return lambda img : self._HLS_thresholding(img, 's')
        elif method == 'h channel':
            return lambda img : self._HLS_thresholding(img,'h')
        elif method == 'l channel':
            return lambda img : self._HLS_thresholding(img,'l')
        else:
            raise ValueError(method)
    def _getSobel(self, img):
        
        gray = img[:,:,0]
        gray = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_DEFAULT)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        
        return sobel_x, sobel_y
    def _getGradient(self,img, output_selector):
        
        sobel_x, sobel_y = self._getSobel(img)

        output = []
        output_selection = ['mag','deg'] if output_selector == 'both' else [output_selector]

        while len(output_selection) > 0:
            if 'mag'in output_selection:
                mag_der = np.sqrt(sobel_x**2 + sobel_y**2)
                mag_der = np.uint8(255*mag_der/np.max(mag_der))

                output_selection.remove('mag')
                output.append(mag_der)
            elif 'dir' in output_selection:
                abs_dir_der = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))

                output_selection.remove('dir')
                output.append(abs_dir_der)
            else:
                raise ValueError(output_selector)

                
        return output

    def _gradient_mag_thresholding(self, img):
        
        mag_grad = self._getGradient(img,'mag')[0]
        interval = self.thresh_intervals['gradient mag']
        
        binary = self._threshold(mag_grad, interval)

        return binary

    
    def _gradient_dir_thresholding(self, img):

        dir_grad = self._getGradient(img,'dir')[0]
        interval = self.thresh_intervals['gradient dir']

        binary = self._threshold(dir_grad, interval)

        return binary
    def _HLS_thresholding(self, img, channel):

        if channel == 'h':
            c = 0
        elif channel == 'l':
            c = 1
        elif channel == 's':
            c = 2
        else:
            raise ValueError(channel)

        img_channel = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,c]
        img_channel = cv2.GaussianBlur(img_channel,(3,3),cv2.BORDER_DEFAULT)
        interval = self.thresh_intervals[channel +' channel']

        binary = self._threshold(img_channel, interval)
        
        return binary

    def _Sobel_thresholding(self, img, coordinate):

        if coordinate == 'x':
            c = 0
        elif coordinate == 'y':
            c = 1
        else:
            raise ValueError(coordinate)

        sobel_img = np.absolute(self._getSobel(img)[c])
        sobel_img = np.uint8(255*sobel_img/np.max(sobel_img))
        interval = self.thresh_intervals['sobel ' + coordinate]

        binary = self._threshold(sobel_img, interval)

        
        return binary
    def combine(self):

        combined_binary_intensity_img = np.zeros(self._image_shape) 
        combined_binary_gradient_img = np.zeros(self._image_shape)
        combined_binary_sobel_img = np.zeros(self._image_shape)

        combined_binary_intensity_img = np.logical_and.reduce(self.thresholded_intensity_imgs)
        combined_binary_gradient_img = np.logical_and.reduce(self.thresholded_gradient_imgs)
        combined_binary_sobel_img = np.logical_and.reduce(self.thresholded_sobel_imgs)
        
        del self.thresholded_intensity_imgs[:]
        del self.thresholded_gradient_imgs[:]
        del self.thresholded_sobel_imgs[:]
        
        return np.logical_or.reduce(\
                    (combined_binary_intensity_img,\
                    combined_binary_gradient_img,\
                    combined_binary_sobel_img\
                    )\
                    ).astype(np.uint8)
