## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/camera_calibration.png "Calibration"
[image2]: ./examples/undistorted.png "Undistorted"
[image3]: ./examples/thresholding.png "Binary Example"
[image4]: ./examples/warp.png "Warp Example"
[image5]: ./examples/slidingWindow.png "Fit Visual"
[image6]: ./examples/output.png "Output"
[video1]: ./video_out.mp4 "Video"


### Camera Calibration

Camera calibration is used to correct the image distortion introduced from the camera sensor.
The calibration procedure has been implemented within the class `CameraCalibrator`, in the file "CameraCalibration.py".
The calibration procedure is based on the analysis of a set of chessboard images. Given the folder containing all the images, the CameraCalibrator for each image find the corners, that are points on the image. Corners are defined as intersection between two edges. They are detected, after converting the color input image to grayscale, using the OpenCv function `cv2.findChessboardCorners`. At each set of corners image points, a set of object 3D points in world coordinates are associated. The object points are distributed on grid without distortion. The camera calibration is performed using the function `cv2.calibrateCamera` that allows to compute the distortion coefficients and the transformation matrix needed to transform 3D points to 2D points. These computed values are needed to perform distortion correction.

![alt text][image1]

### Distortion correction

Once the calibration is performed, it's possible to correct the distortion introduced by the camera using the CameraCalibrator class method `applyCalibration` that is based on the OpenCv function `cv2.undistort`

#### 1. Examples of a distortion-corrected images.

![alt text][image2]

#### 2. Use of color transforms and gradients to create a thresholded binary image.

In order to be able to detect lines of different colors under several light condition several thresholding methods have been combined:
	* S channel
	* L channel
	* Gradient magnitude
	* Gradient direction
	* Sobel derivatives

All the above thresholding techniques have been applied using the class `Thresholder` (Thresholding.py file). Before to apply every thresholding, every image has been processed with a gaussian smoothing to remove noise and eliminate small details that are not relevant. 
![alt text][image3]

#### 3. Perspective transform.
After an inspection of the undistorted images, some fixed points has been choosed as source and destination points for the perspective transformation. The source points have been taken in order to form a trapezoid that contains the lane. Destination points, instead have been chosen to form a rectangle.

Source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that with those points my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Lane-line pixels detection and polynomial fit
The class `SlidingWindow` is responsible to find the lane boundary pixels (file SlidingWindow.py). This class, internally, uses two other classes: `PolyFit` (file PolyFitter.py) and `Window` (defined in the same file of `SlidingWindow`). The first class is used as wrapper to the `np.polyfit` function that allow to associate to the detected lane boudaries points two polynomials useful to compute the lane curvature. The second class is used to model the windows that moving on the image detect lane pixels. The `SlidingWindow` class works in three main steps. As first step (`_init` method) it initializes itself computing the histogram of the lower half of the warped image. The histogram is useful to select the starting left and right line starting positions by looking at the max value on the left and right histogram half part. After this first step the windows are slid vertically for both the left and right line and the pixels within the windows are taken as lane boundaries pixels (`lanesPixelsDetection` method). On the found pixels, two second order polynomials are fitted. Finally, if valid polynomials are available from the computations on the previous image then the new lines are searched in an area surrounding the previously detected lines (`updateDetection` method). When valid polynomials are not longer available, the class stars again at the first step.
To verify if the detected lines are still valid, the method `sanityCheck` of the class `ImageProcessor` is used.

![alt text][image5]

#### 5. Radius of curvature of the lane and position of the vehicle with respect to center computation.

The rasius of curvature is computed according by the function `computeCurvature` of the class `Line` for each line (Line.py file). The final value provided for the curvature is the average between the two compute curvatures (`getCurvature` method, `ImageProcessor` class). The method `computeVehiclePosition` of the class `ImageProcessor` computes the position of the vechicle with respect to the center of the image computing the mean x position of the detected lines and subtracting it to the image x middle point.

#### 6. Example image of result plotted back down onto the road

`applyResult` method of the class `ImageProcessor` is responsible to show the final result on the input undistorted image.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### Used techniques to reach better results

#### 1. Smoothing over time

The lane detected for each frame is an average of the current detection and the previous 10 frames. In this way, suddenly changes due to noise on the lane estimation are avoided. The lines pixels are smoothed over time as well.

#### 2. Discard bad frames

In the case that also after the reinitialization of the sliding window algorithm, the detected lane is not considered feasible, the current lane estimation is discarded and the previous best estimation is used instead.

#### Possible improvements

Currently the algorithm is quite slow so it's not possible to use it in real time. It could be possible to divide the image in a left and right side and assign each side to a different thread or process.
Furthermore, the presented pipeline could fail if there is other road signs on the lane different from simple lines.
Finally, the algorithm should take into account the obstruction of the lane boundaries from big objects, like other vehicles. For instance an overtake phase could be a failure situation.

