import numpy as np
import cv2
import matplotlib.pyplot as plt

class Thresholding(object):

    def __init__(self):
        self.thresholded_intensity_imgs = []
        self.thresholded_gradient_imgs = []
        self.thresh_intervals = {}
        
        self.current_img = None
        self.current_gradient = []
        self._image_shape = None

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
        if method == 's channel':
            self.thresholded_intensity_imgs.append(thresholding_fcn(rgb_img))
        else:
            self.thresholded_gradient_imgs.append(thresholding_fcn(rgb_img))

    def _getThresholdingMethod(self, method):

        if method == 'gradient mag':
            return self._gradient_mag_thresholding
        elif method == 'gradient dir':
            return self._gradient_dir_thresholding
        elif method == 's channel':
            return lambda img : self._HLS_thresholding(img, 's')
        else:
            raise ValueError(method)
    
    def _getGradient(self,img, output_selector):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
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
        print("SHAPE {}".format(img_channel.shape))
        interval = self.thresh_intervals[channel +' channel']

        binary = self._threshold(img_channel, interval)
        
        return binary

    def combine(self):

        combined_binary_intensity_img = np.zeros(self._image_shape) 
        combined_binary_gradient_img = np.zeros(self._image_shape)

        for img in self.thresholded_intensity_imgs:
            combined_binary_intensity_img[(img == 1)] = 1

        for img in self.thresholded_gradient_imgs:
            combined_binary_gradient_img[(img == 1)] = 1
        
        del self.thresholded_intensity_imgs[:]
        del self.thresholded_gradient_imgs[:]
        
        """plt.figure()
        plt.imshow(combined_binary_intensity_img)
        plt.show()
        plt.figure()
        plt.imshow(combined_binary_gradient_img)
        plt.show()
        """
        return np.logical_and(\
                    combined_binary_intensity_img,\
                    combined_binary_gradient_img\
                    ).astype(np.uint8)
