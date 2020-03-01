import numpy as np
import cv2
from utilities import getNonZero
from PolyFitter import PolyFit
from utilities import Visualizer
import matplotlib.pyplot as plt

class SlidingWindow(object):
    """
    Sliding Window for lines finding
    """
    def __init__(self, n_windows=9, margin=100, min_pix_number=50):

        # PolyFit class used to estimate the polynomial that
        # better represent the left and the right line
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
        
        self.leftx_pixels = None
        self.lefty_pixels = None
        self.rightx_pixels = None
        self.righty_pixels = None
    
    def updateLines(self, left_line, right_line):
        # Given two Line classes, they are filled with the points and polynomials found

        # Store if lines are found
        left_line.detected, right_line.detected = [self.left_fitter.isValid(), self.right_fitter.isValid()]
        
        # Save the polynomials
        left_line.setFit(self.left_fitter.getCurrentFit())
        right_line.setFit(self.right_fitter.getCurrentFit())
        
        # Save the x points belonging to the lines
        left_line.recent_xfitted.append(self.left_fitter.getFittedData()[0])
        right_line.recent_xfitted.append(self.right_fitter.getFittedData()[0])
        # Save the y points belonging to the lines
        left_line.ydata = self.left_fitter.getFittedData()[1]
        right_line.ydata = self.right_fitter.getFittedData()[1]

        # Save all the pixels used for the polynomial estimation
        left_line.allx, right_line.allx = [self.leftx_pixels, self.rightx_pixels]
        left_line.ally, right_line.ally = [self.righty_pixels, self.righty_pixels]
        
        # Update the lines with the new injected values
        left_line.update()
        right_line.update()

        return
        


    def getCurrentFit(self):

        return self.left_fitter.getCurrentFit(), self.right_fitter.getCurrentFit()

    def resetFitters(self):
        # reset the PolyFit classes and re-initialize the slidind window search
        self.is_initialized = False
        self.left_fitter.invalidate()
        self.right_fitter.invalidate()
    
    def find(self, img):
        # find the lines
        
        if not self.is_initialized:
            self._init(img)

        if not self.left_fitter.isValid() or not self.right_fitter.isValid():
            # LanesPixelsDetection tries to find the lines moving the sliding windows over the image
            leftx, lefty, rightx, righty = self.lanesPixelsDetection(img)
        else: # if a valid polynomial representation of the lines is present then
              # search the new lines arround the location containing the previous lines
            leftx, lefty, rightx, righty = self.updateDetection(img)
        
        # blank image used later to store the lines points
        out_img = np.zeros((img.shape[0], img.shape[1], 3))
        
        if leftx.size and lefty.size and rightx.size and righty.size:
            # Fit the data
            ret = True

            pol_left = self.left_fitter.fit(lefty, leftx)
            pol_right = self.right_fitter.fit(righty, rightx)

            self.left_fitter.fillPolyData((0, img.shape[0]))
            self.right_fitter.fillPolyData((0, img.shape[0]))
        
            self.leftx_pixels = leftx
            self.lefty_pixels = lefty
            self.rightx_pixels = rightx
            self.righty_pixels = righty
        
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [255, 0, 0]

        else:
            ret = False
        return out_img, ret


    def updateDetection(self, img):
        # update the lines detection searching in an area close to the previous found lines
        nonzero_y, nonzero_x = getNonZero(img)

        left_lane_inds = (nonzero_x > (self.left_fitter.eval(nonzero_y)-self.margin)) &\
                (nonzero_x < (self.left_fitter.eval(nonzero_y)+self.margin))
        right_lane_inds = (nonzero_x > (self.right_fitter.eval(nonzero_y)-self.margin)) &\
                (nonzero_x < (self.right_fitter.eval(nonzero_y)+self.margin))
        
        leftx = nonzero_x[left_lane_inds]
        lefty = nonzero_y[left_lane_inds]
        rightx = nonzero_x[right_lane_inds]
        righty = nonzero_y[right_lane_inds]

        return leftx, lefty, rightx, righty
        
    def lanesPixelsDetection(self, img):
        # search lines pixels using a sliding window from the bottom of the image to the top

        nonzero_y, nonzero_x = getNonZero(img)
        
        all_left_lane_inds = []
        all_right_lane_inds = []

        for window in range(self.n_windows):

            left_lane_inds = self.left_window.getIndsInWindow(\
                    nonzero_x,\
                    nonzero_y\
                    ).nonzero()[0]
            right_lane_inds= self.right_window.getIndsInWindow(\
                    nonzero_x,\
                    nonzero_y\
                    ).nonzero()[0]

            all_left_lane_inds.append(left_lane_inds)
            all_right_lane_inds.append(right_lane_inds)
            
            leftx_current = None
            rightx_current = None
            
            if len(left_lane_inds) > self.min_pix_number:
                leftx_current = np.int(np.mean(nonzero_x[left_lane_inds]))
            if len(right_lane_inds) > self.min_pix_number:
                rightx_current = np.int(np.mean(nonzero_x[right_lane_inds]))
            
            self.left_window.positionStep(leftx_current)
            self.right_window.positionStep(rightx_current)


        all_left_lane_inds = np.concatenate(all_left_lane_inds)
        all_right_lane_inds = np.concatenate(all_right_lane_inds)

        return nonzero_x[all_left_lane_inds],\
                nonzero_y[all_left_lane_inds],\
                nonzero_x[all_right_lane_inds],\
                nonzero_y[all_right_lane_inds]


    def _init(self, img):

        # Algorithm initialization
        
        # The histogram is computed in order to find the starting points of the lines at the bottom
        # of the image
        self.img_shape = img.shape
        histogram = np.sum(img[self.img_shape[0]//2:,:], axis=0)
        self.mid_point = histogram.shape[0]//2

        self.leftx_base = np.argmax(histogram[:self.mid_point])
        self.rightx_base = np.argmax(histogram[self.mid_point:]) + self.mid_point

        self.window_height = np.int(self.img_shape[0]//self.n_windows)
        
        self.left_window = Window(self.leftx_base,\
                self.img_shape[0]-self.window_height//2,\
                self.margin,\
                self.window_height)
        self.right_window = Window(self.rightx_base,\
                self.img_shape[0]-self.window_height//2,\
                self.margin,\
                self.window_height)

        self.is_initialized = True
    
    def visualizeUpdate(self, out_img):
        
        window_img = np.zeros_like(out_img)

        leftx, ploty = self.left_fitter.getFittedData()
        rightx, _ = self.right_fitter.getFittedData()
 
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([leftx-self.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([leftx+self.margin,
                                ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([rightx-self.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([rightx+self.margin,
                                ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
 
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
 
        plt.figure()
        plt.imshow(result)
         # Plot the polynomial lines onto the imagself.e
        plt.plot(leftx, ploty, color='yellow')
        plt.plot(rightx, ploty, color='yellow')
        plt.show()

class Window(object):

    """
        (x1,y2)-----(x2,y2)
          |            |
          |            |
          |            |
        (x1,y1)-----(x2,y1)
    """

    """
    Model of a window that can moves horizontally and vertically on a 2D plane
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
        
        self.y1 = self.opposite_sum(y0, self.h//2)

        self.y2 = self.sum(self.y1, self.h)
        self.x2 = self.x1 + self.w

    def showOnImage(self, img, color=255, width=2):

        out_img = np.copy(img)
        cv2.rectangle(out_img, (self.x1, self.y1), (self.x2, self.y2), color, width)
        Visualizer.show(img)
        Visualizer.show(out_img)
        return out_img

    def positionStep(self, x0):
        
        self.moveVertically()
        if x0 is not None:
             self.updateCenterX(x0)

    def moveVertically(self):
        
        self.y1 = self.y2
        self.y2 = self.sum(self.y1, self.h)

    def updateCenterX(self, x0):
        self.x1 = x0 - self.w//2
        self.x2 = x0 + self.w//2

    def getIndsInWindow(self, inds_x, inds_y):
        
        inds = ((inds_y >= self.y2) &\
                (inds_y < self.y1) &\
                (inds_x >= self.x1) &\
                (inds_x < self.x2))
        return inds
