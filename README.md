## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

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

[image1a]: ./camera_cal/calibration2.jpg "Original"
[image1b]: ./output_images/undistort_chessboard.png "Undistorted"
[image2a]: ./test_images/test3.jpg "Road Transformed"
[image2b]: ./output_images/distortion_corrected.jpg "Road Transformed"
[image3]: ./output_images/binary_test1.jpg "Binary Example"
[image4a]: ./output_images/outlined_straight_lines2.jpg "Before Warp"
[image4b]: ./output_images/warped_straight_lines2.jpg "After Warp"
[image5a]: ./examples/color_fit_lines.jpg "Fit Visual"
[image5b]: ./output_images/lane_fitting.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

This is the writeup and so you're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in the Python file `camera_calibration.py`. Initializing an object of class `Calibration`
automatically determines the chessboard corners (in the function `_find_corners`, lines 31-71) and calibrates the camera (in 
the function `_calibrate_camera`, lines 82-92).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for 
each calibration image.  Thus, `objp` (`camera_calibration.py`, lines 41-42) is just a replicated array of coordinates, 
and `objpoints` (`camera_calibration.py`, line 60) will be appended with a copy of it every time I successfully detect 
all chessboard corners in a test image.  `imgpoints` (`camera_calibration.py`, line 63) will be appended with the 
(x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion 
coefficients using the `cv2.calibrateCamera()` function (`camera_calibration.py`, line 89).

I applied this distortion correction to the test image (`calibration2.jpg`) using the `cv2.undistort()` function 
(`camera_calibration.py`, line 104) and obtained this result: 

![alt text][image1a]  ![alt text][image1b]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I applied the distortion correction to one of the test images 
like this one (`./test_examples/test3.jpg`):

![alt text][image2a]

The way I structured the code, one just needs to call the `undistort` member function in a `Calibration` object on the
image to undistort the image. This is exactly what we do in `lane_detect.py`. Here we instantiated an object of
`Calibration` called `calibration` (`lane_detect.py`, line 44), and use it to undistort the image above 
(`lane_detect.py`, line 84).

![alt text][image2b]

We can see the undistorted image is similar to the original but is, for example, missing the tree shadow on the
bottom left corner of the original, as well as the last bit of lane marking on the bottom right of the original.

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  
I used a combination of color and gradient thresholds to generate a binary image (see `lane_detect.py`, function 
`binary_filter`, lines 175-210, and `edge_detection.py`, the functions `abs_sobel_thresh`, lines 8-26, 
`mag_thresh`. lines 28-45, `dir_thresh`, lines 47-64). The functions `abs_sobel_thresh`, `mag_thresh`, and
 `dir_thresh` correspond to absolute thresholding (thresholding on derivative of x or y), magnitude thresholding 
 (thresholding on the magnitude of the gradient), and direction thresholding (thresholding on the angle/direction).

Finally, we apply these functions, together with a color (or saturation) threshold (`lane_detect.py`, lines 201-202)
to produce a threshold binary image like the example below.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the function `lane_detect`, lines 102-106, in the file `lane_detection.py`.
The perspective transform is performed using the functions `cv2.getPerspectiveTransform` and `cv2.warpPerspective`. 
The function `cv2.getPerspectiveTransform` calculates the transformation matrix using the source 
(`PERSPECTIVE_SRC_COORD`) and destination (`PERSPECTIVE_DST_COORD`) points, and `cv2.warpPerspective` performs 
the actual transformation.  

I calculated the source (`PERSPECTIVE_SRC_COORD`) and destination (`PERSPECTIVE_DST_COORD`) manually on 
`straight_lines1.jpg` and `straight_lines2.jpg`. They are hardcoded in `lane_detection.py` in lines 21-29 and are
also provided here for reference:

# Global constant calibrated using straight_lines1.jpg and straight_lines2.jpg
```
PERSPECTIVE_SRC_COORD = np.float32([[218.70, 710],
                                    [595.60, 450],
                                    [685.10, 450],
                                    [1087.30, 710]])

PERSPECTIVE_DST_COORD = np.float32([[300, 720],
                                    [300, 0],
                                    [1000, 0],
                                    [1000, 720]])
```
This resulted in the following source and destination points:

| Source           | Destination   | 
|:----------------:|:-------------:| 
| 595.60, 450      | 300, 0        | 
| 218.70, 710      | 300, 720      |
| 1087.30, 710     | 1000, 720     |
| 685.10, 450      | 1000, 0       |

I verified that my perspective transform was working as expected by drawing the `PERSPECTIVE_SRC_COORD` and 
`PERSPECTIVE_DST_COORD` points onto a test image and its warped counterpart to verify that the lines appear 
parallel in the warped image.

![alt text][image4a]
![alt text][image4b]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane fitting algorithm is contained in the file `lane_fitting.py`, where I defined a `QuadraticLaneFitter` class to
encapsulate all the steps applied in fitting the lanes. Here is an outline of the steps:
1. Apply sliding window to identify pixels that make up the left and right lanes 
(`lane_fitting.py`, lines 124-127, and 335-394).
2. Fit a quadratic polynomial on the left lane and one on the right lane (`lane_fitting.py`, lines 142-156), save
the fit and the lane pixels.
3. Save the curvatures (lines 155-156), the median distance between the left and right lane pixels (lines 172-173),
and the offset from the center (lines 175-177).
4. Once we have fitted a few frames (specified by the `warmup` parameter) following steps 1-3, we replace the 
sliding window search with a fixed window around the previously fitted lanes (`lane_fitting.py`, line 129, and in 
function `_fix_window`, lines 317-342). If there aren't enough lane pixels, then we revert back to sliding window
search.
5. The lane fitting algorithm also changes once we have fitted a few frames&mdash;since we can now rely on past 
information on lane widths and lane locations to calibrate lane locations in the current frame. The lane calibration 
algorithm is encapsulated in the function `_calibrate_fit` (`lane_fitting.py`, lines 240-297). The approach is to first 
fit a quadratic polynomial on the lane pixels using `np.polyfit`, and determine the fitted lane pixels. We then identify
where the fitted lanes are too wide or too narrow compared with the median of the lane widths from the last 20
frames. For these pixels, we check whether the pixel in the left lane or the right lane deviated the most (in the
horizontal direction) from their previous counterparts (which we take to be the mean of the left/right lane pixel 
with the same y-coordinate from the previous 5 frames, see lines 226-227 of `lane_fitting.py`), and adjust it
accordingly (by averaging it with the previous pixel, see lines 279-289 of `lane_fitting.py`). We repeat this 
procedure until the width of the current fitted lane is within +/- 10 pixels of the median of the widths 
of the last 20 frames. Finally we refit on quadratic polynomial on the adjusted pixels (lines 291-296 of 
`lane_fitting.py`). We refer the reader to `lane_fitting.py` for more details on the implementation.
6. Finally, the actual fitted lane pixels returned is the weighted average of the previous 5 frames (`lane_fitting.py`,
function `left_fitx`, lines 38-51, and function `right_fitx`, lines 53-66).

Example of quadratic fitted lanes:

![alt text][image5a] 

Example of fixed window around lane pixels based on previous detections:

![alt text][image5b]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I used the algorithm provided in the course to calculate the lane curvatures (for both the left and right lanes). Please 
see `lane_fitting.py`, function `_find_curvature`, lines 299-315. The radius of curvature shown in the image below is the 
average of that of the left and right lanes (rounded to two decimal places).

The offset from center (or position of the vehicle) is calculated as the difference between the middle of the image and 
the middle of the bottom of the two fitted lane lines, scaled by 3.7/700 to convert the units from pixels to meters. 
This calculation is coded in `lane_fitting.py`, lines 177-179.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The coloring of the detected lane is implemented in the function `draw_lanes` on lines 143-175 of the file 
`lane_detection.py`. The texts with radius of curvature and vehicle positions are added on lines 131-140 of the same
file. Finally, the saving the image is performed on lines 61-68 of the same file.

Processing a video file is essentially piecing together a series of these images. This is done on lines 50-59 of the file
`lane_detection.py`

Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have already discussed my approach to identifying the lane lines. Some of the challenges/issues I 
encountered when working on the project:
1. The color and gradient thresholding is done by trial-and-error. A more systematic approach based on the typical
range of gradient, saturation, or some other property value of lane lines is better. 
2. Along the same lines, I thought about applying a Gaussian filter to reduce the noise created by shadow, 
tire/skid marks, grass on the side of the road, etc. But I am a novice at computer image processing/signal processing
so I decided to save this for later when I have a better understanding of image processing.
3. Perspective transform causes loss in sharpness in the top part of the warped binary images.
4. The lane detection algorithm is expected to run in real-time, and needs to keep up with the vehicle speed if it
is to be practical in aiding self-driving. Our code is written in Python, which is not the ideal language if
execution performance is a concern. 
5. I thought about applying a convolutional/recurrent neural network to predict the lane pixel coordinates. This
may very well work, but there isn't enough data of sufficient variety to train on. Potentially if we have a good
lane detection algorithm that predicts lane lines accurately, then we can use it to generate training data for 
such a neural network. I hope to try this out in the future.
6. An underlying assumption in my algorithm is that the lanes are of constant width. While this is true in general,
there are situations where the lane lines change width.
7. Moreover, there are special road situations (e.g. roadwork with cone-identified lanes) under which this algorithm 
will fail.

Besides the `project_video.mp4`, I also applied my algorithm to the `challenge_video.py` and 
`harder_challenge_video.py`. Unfortunately the algorithm did not perform well on these two videos. There is too much 
noise in these videos and that confused the algorithm.

Where will my pipeline likely fail? Clearly the pipeline requires the lanes to be properly detected in the warmup 
window (the initial 20 frames), otherwise the future lane dectection will be impacted. The pipeline is also 
most prone to failure in this warmup period as it requires at least 50 pixels identified for each lane line in
order to fit the lanes. So if any of the first 20 frames do have have enough pixels (take an extreme case, 
say the frame is all black), then the pipeline will throw an exception. 

What could I do to make it more robust? We can adjust the warmup window to just 1 frame (and also add some 
exception handling), or simple adjust the `minpix` parameter to accept fewer pixels, but a better solution maybe to 
just preload the algorithm with some good footage so it "warms up". In the end, the dependency on the quality of
the footage cannot be replaced by robustness and error-handling in the sofware alone.