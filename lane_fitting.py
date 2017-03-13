# -*- coding: utf-8 -*-

__author__ = 'Johnson Jia'

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Quadratic_lane_fitter():
    """ Apply sliding window filtering of the lane lines; fit quadratic
    polynomials to model the lane lines.

    Attributes:
        left_fit: Coefficients of the quadratic fit for the left lane.
        right_fit: Coefficients of the quadratic fit for the right lane.
    """

    def __init__(self, nwindows=9, margin=100, minpix=50):
        self._nwindows = nwindows
        self._margin = margin
        self._minpix = minpix
        self._left_fit = None
        self._right_fit = None

    @property
    def left_fit(self):
        return self._left_fit

    @property
    def right_fit(self):
        return self._right_fit

    def find_lanes(self, image, display=False):
        """ Determine the quadratic polynomial fits.

        Args:
            image: The image with the lane lines.
            display: Display the image with lane lines drawn.
        """
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # If we have not found a reasonable window around the lanes.
        if self._left_fit is None:
            l_lane_inds, r_lane_inds, out_img = self._sliding_window(image)
        else:
            l_lane_inds, r_lane_inds = self._fix_window(image)
            out_img = np.dstack((image, image, image)) * 255

        # Extract left and right line pixel positions
        leftx = nonzerox[l_lane_inds]
        lefty = nonzeroy[l_lane_inds]
        rightx = nonzerox[r_lane_inds]
        righty = nonzeroy[r_lane_inds]

        # Fit a second order polynomial to each
        self._left_fit = np.polyfit(lefty, leftx, 2)
        self._right_fit = np.polyfit(righty, rightx, 2)


        if display:
            # Generate x and y values for plotting
            ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
            left_fitx = self._left_fit[0] * ploty**2 + \
                        self._left_fit[1] * ploty + self._left_fit[2]
            right_fitx = self._right_fit[0] * ploty**2 + \
                         self._right_fit[1] * ploty + self._right_fit[2]
            out_img[nonzeroy[l_lane_inds], nonzerox[l_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[r_lane_inds], nonzerox[r_lane_inds]] = [0, 0, 255]
            plt.figure()
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()

    def _fix_window(self, image):
        """ Find lane indicators using previous fitted window.

        Args:
            image: The image in question.

        Returns:
            The left lane and right lane indicators or window boundaries.
        """
        # Identify the x and y positions of all nonzero pixels in the image.
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (self._left_fit[0] * (nonzeroy**2) +
                                       self._left_fit[1] * nonzeroy +
                                       self._left_fit[2] - self._margin)) &
                          (nonzerox < (self._left_fit[0] * (nonzeroy**2) +
                                       self._left_fit[1] * nonzeroy +
                                       self._left_fit[2] + self._margin)))
        right_lane_inds = ((nonzerox > (self._right_fit[0] * (nonzeroy**2) +
                                        self._right_fit[1] * nonzeroy +
                                        self._right_fit[2] - self._margin)) &
                           (nonzerox < (self._right_fit[0] * (nonzeroy ** 2) +
                                        self._right_fit[1] * nonzeroy +
                                        self._right_fit[2] + self._margin)))
        return left_lane_inds, right_lane_inds

    def _sliding_window(self, image):
        """ Apply sliding windows to detect lane lines.

        Args:
            image: The image in question.

        Returns:
            The left lane and right lane indicators or window boundaries.
            also the image with the windows overlayed.
        """
        histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
        out_img = np.dstack((image, image, image)) * 255
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows.
        window_height = np.int(image.shape[0]//self._nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self._nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window+1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_xleft_low = leftx_current - self._margin
            win_xleft_high = leftx_current + self._margin
            win_xright_low = rightx_current - self._margin
            win_xright_high = rightx_current + self._margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) &
                              (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) &
                               (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean
            # position
            if len(good_left_inds) > self._minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self._minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        return left_lane_inds, right_lane_inds, out_img
