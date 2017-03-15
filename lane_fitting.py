# -*- coding: utf-8 -*-

__author__ = 'Johnson Jia'

import cv2
import numpy as np
import matplotlib.pyplot as plt


class QuadraticLaneFitter():
    """ Apply sliding window filtering of the lane lines; fit quadratic
    polynomials to model the lane lines.

    Attributes:
        left_fitx: x-coordinates of the quadratic fitted left lane.
        right_fitx: x-coordinates of the quadratic fitted right lane.
        ploty: y-coordinates of the quadratic fitted lanes.
    """

    def __init__(self, nwindows=9, margin=100, minpix=50):
        self._nwindows = nwindows
        self._margin = margin
        self._minpix = minpix
        self._left_fitx = None
        self._right_fitx = None
        self._left_curvature = None
        self._right_curvature = None
        self._ploty = None

    @property
    def left_fitx(self):
        """
        Getter function for the x coordinates of the left lane.

        Returns:
            The x coordinates of the left lane weighted averaged over
            the last five frames.
        """
        if len(self._left_fitx) > 0:
            return np.average(self._left_fitx[-1:-6:-1], axis=0,
                              weights=np.arange(min(len(self._left_fitx),5))+1)
        else:
            return None

    @property
    def right_fitx(self):
        """
        Getter function for the x coordinates of the right lane.

        Returns:
            The x coordinates of the right lane weighted averaged over
            the last five frames.
        """
        if len(self._right_fitx) > 0:
            return np.average(self._right_fitx[-1:-5:-1], axis=0,
                              weights=np.arange(min(len(self._left_fitx),5))+1)
        else:
            return None

    @property
    def ploty(self):
        """
        Getter function for the y coordinates of both lanes.

        Returns:
            The y coordinates of both lanes. The coordinates only
            depend on the size of the image.
        """
        return self._ploty

    @property
    def left_curvature(self):
        """

        Returns:

        """
        return None

    @property
    def right_curvature(self):
        """

        Returns:

        """
        return None

    def find_lanes(self, image):
        """ Determine the quadratic polynomial fits.

        Args:
            image: The image with the lane lines.
        """
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # If we have not found a reasonable window around the lanes.
        if self._left_fitx is None:
            self._left_fitx = []
            self._right_fitx = []
            l_lane_inds, r_lane_inds = self._sliding_window(image)
        else:
            l_lane_inds, r_lane_inds = self._fix_window(image)

        # Extract left and right line pixel positions.
        leftx = nonzerox[l_lane_inds]
        lefty = nonzeroy[l_lane_inds]
        rightx = nonzerox[r_lane_inds]
        righty = nonzeroy[r_lane_inds]

        # Use the previous fitted lane points if the new points deviate
        # too much from them.
        if len(self._left_fitx) > 0 and \
           np.mean(leftx) - np.mean(self._left_fitx[-1]) > self._margin/2:
            self._left_fitx.append(self._left_fit[-1])
        elif len(self._right_fitx) > 0 and \
             np.mean(rightx) - np.mean(self._right_fitx[-1]) > self._margin/2:
            self._right_fitx.append(self._right_fitx[-1])
        else:
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            # Generate x and y values for plotting
            self._ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
            self._left_fitx.append(left_fit[0] * np.square(self._ploty) + \
                                   left_fit[1] * self._ploty + left_fit[2])
            self._right_fitx.append(right_fit[0] * np.square(self._ploty) + \
                                    right_fit[1] * self._ploty + right_fit[2])

    def _set_curvature(self):
        pass

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
        """
        histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
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
        return left_lane_inds, right_lane_inds