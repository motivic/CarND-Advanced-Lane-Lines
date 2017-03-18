##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image4a]: ./output_images/undistorted_straight_lines2.jpg "Before Warp"
[image4b]: ./output_images/warped_straight_lines2.jpg "After Warp"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

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
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one 
(`./test_examples/test3.jpg`):

![alt text][image2a]

The way I structured the code, one just needs to call the `undistort` member function in a `Calibration` object on the
image to undistort the image. This is exactly what we do in `lane_detect.py`. Here we instantiated an object of
`Calibration` called `calibration` (`lane_detect.py`, line 44), and use it to undistort the image above 
(`lane_detect.py`, line 84).

![alt text][image2b]

We can see the undistorted image is similar to the original but is, for example, missing the tree shadow on the
bottom left corner of the original, and also the last bit of lane marking on the bottom right of the original.

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
The function `cv2.getPerspectiveTransform` calculates the transformation matrix using the source (`src`) and 
destination (`dst`) points, and `cv2.warpPerspective` performs the actual transformation.  

I calculated the source (`PERSPECTIVE_SRC_COORD`) and destination (`PERSPECTIVE_DST_COORD`) manually using 
`straight_lines1.jpg` and `straight_lines2.jpg`. They are hardcoded in `lane_detection.py` in lines 21-29. 
chose the hardcode the source and destination points in the following manner:

# Global constant calibrated using straight_lines1.jpg
# and straight_lines2.jpg
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

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

