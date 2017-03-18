# -*- coding: utf-8 -*-

__author__ = 'Johnson Jia'

import cv2
import matplotlib.pyplot as plt
import numpy as np

class QuadraticLaneFitter():
    """ Apply sliding window filtering of the lane lines; fit quadratic
    polynomials to model the lane lines.

    Attributes:
        left_fitx: x-coordinates of the quadratic fitted left lane.
        right_fitx: x-coordinates of the quadratic fitted right lane.
        ploty: y-coordinates of the quadratic fitted lanes.
    """

    def __init__(self, nwindows=9, margin=100, minpix=50, warmup = 20,
                 ym_per_pix = 30/720, xm_per_pix = 3.7/700):
        # Define conversions in x and y from pixels space to meters
        self._ym_per_pix = ym_per_pix # meters per pixel in y dimension
        self._xm_per_pix = xm_per_pix # meters per pixel in x dimension
        self._nwindows = nwindows
        self._margin = margin
        self._minpix = minpix
        self._warmup = warmup
        self._lane_widths = []
        self._left_fit = None
        self._right_fit = None
        self._left_fitx = None
        self._right_fitx = None
        self._left_curverad = None
        self._right_curverad = None
        self._offset = None
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
            return np.average(self._right_fitx[-1:-6:-1], axis=0,
                              weights=np.arange(min(len(self._left_fitx),5))+1)
        else:
            return None

    @property
    def ploty(self):
        """ Getter function for the y coordinates of both lanes.

        Returns:
            The y coordinates of both lanes. The coordinates only
            depend on the size of the image.
        """
        return self._ploty

    @property
    def left_curvature(self):
        """ Getter function for the curvature of the left lane.

        Returns:
            The curvature of the left lane.
        """
        return self._left_curverad

    @property
    def right_curvature(self):
        """ Getter function for the curvature of the right lane.

        Returns:
            The curvature of the right lane.
        """
        return self._right_curverad

    @property
    def offset(self):
        """ Getter function for the offset from the center of the lane.

        Returns:
            The offset from the center of the lane.
        """
        return self._offset

    def find_lanes(self, img, trim_top=0.1, visualize=False):
        """ Determine the quadratic polynomial fits.

        Args:
            img: The image with the lane lines.
            trim_top: The percentage of the top of the image to trim away.
            visualize: If true, return image; otherwise return None.

        Returns:
            If visualize is true, then return the image with
            lane pixels colored.
        """
        # Trim the top of the image according trim_top.
        ydim, xdim = img.shape
        image = img[int(ydim*trim_top):, :]
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
            if l_lane_inds.size < self._minpix/2 or \
               r_lane_inds.size < self._minpix/2:
                l_lane_inds, r_lane_inds = self._sliding_window(image)

        # Extract left and right line pixel positions.
        leftx = nonzerox[l_lane_inds]
        lefty = nonzeroy[l_lane_inds]
        rightx = nonzerox[r_lane_inds]
        righty = nonzeroy[r_lane_inds]

        # Use the previous fitted lane points if the new points deviate
        # too much from them.
        if self._warmup > 0:
            self._warmup -= 1
            # Generate x and y values for plotting.
            self._ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
            # Fit quadratic polynomials to lane lines.
            self._left_fit = np.polyfit(lefty, leftx, 2)
            self._right_fit = np.polyfit(righty, rightx, 2)
            self._left_fitx.append(self._left_fit[0]*np.square(self._ploty) +
                                   self._left_fit[1]*self._ploty +
                                   self._left_fit[2])
            self._right_fitx.append(self._right_fit[0]*np.square(self._ploty) +
                                    self._right_fit[1]*self._ploty +
                                    self._right_fit[2])
            self._left_curverad = self._find_curvature(self._left_fitx[-1])
            self._right_curverad = self._find_curvature(self._right_fitx[-1])
        else:
            lfit, rfit, lfitx, rfitx = self._calibrate_fit(leftx, lefty,
                                                           rightx, righty)
            if np.max(abs(lfitx - self._left_fitx[-1])) > self._margin/3:
                self._left_fitx.append(self._left_fitx[-1])
            else:
                self._left_fit = lfit
                self._left_fitx.append(lfitx)
                self._left_curverad = self._find_curvature(lfitx)
            if np.max(abs(rfitx - self._right_fitx[-1])) > self._margin/3:
                self._right_fitx.append(self._right_fitx[-1])
            else:
                self._right_fit = rfit
                self._right_fitx.append(rfitx)
                self._right_curverad = self._find_curvature(rfitx)
        self._lane_widths.append(np.median(self._right_fitx[-1] - \
                                           self._left_fitx[-1]))
        # Calculate offset
        self._offset = (xdim/2 - \
                        (self._right_fitx[-1][-1] + \
                         self._left_fitx[-1][-1])/2) * self._xm_per_pix

        if visualize:
            return self._visualize(img, leftx, lefty, rightx, righty)
        else:
            return None

    def _visualize(self, img, lx, ly, rx, ry):
        """ Color the identified lane pixels.

        Args:
            img: The original image.
            lx: X-coordinates of the pixels for the left lane.
            ly: Y-coordinates of the pixels for the left lane.
            rx: X-coordinates of the pixels for the right lane.
            ry: Y-coordinates of the pixels for the right lane.

        Returns:
            The image with lane pixels colored.
        """
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((img, img, img)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[ly, lx] = [255, 0, 0]
        out_img[ry, rx] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array(
            [np.transpose(
                np.vstack([self._left_fitx[-1] - self._margin, self._ploty]))
            ])
        left_line_window2 = np.array(
            [np.flipud(np.transpose(
                np.vstack([self._left_fitx[-1] + self._margin, self._ploty])))
            ])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(
                np.vstack([self._right_fitx[-1] - self._margin, self._ploty]))
            ])
        right_line_window2 = np.array(
            [np.flipud(np.transpose(
                np.vstack([self._right_fitx[-1] + self._margin, self._ploty])))

            ])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(self._left_fitx[-1], self._ploty, color='yellow')
        plt.plot(self._right_fitx[-1], self._ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig('tmp.jpg')
        return plt.imread('tmp.jpg')

    def _calibrate_fit(self, lx, ly, rx, ry):
        """ Check whether the lane points discovered are reasonable.

        Args:
            lx: X-coordinates of the pixels for the left lane.
            ly: Y-coordinates of the pixels for the left lane.
            rx: X-coordinates of the pixels for the right lane.
            ry: Y-coordinates of the pixels for the right lane.

        Returns:
            The fit and the coordinates of the fitted lanes.
        """
        try:
            lfit = np.polyfit(ly, lx, 2)
        except TypeError:
            lfit = self._left_fit
        try:
            rfit = np.polyfit(ry, rx, 2)
        except TypeError:
            rfit = self._right_fit
        lfitx = lfit[0]*np.square(self._ploty) + \
                lfit[1]*self._ploty + lfit[2]
        rfitx = rfit[0]*np.square(self._ploty) + \
                rfit[1]*self._ploty + rfit[2]
        # Check the lane separation is reasonable.
        lane_width = np.median(self._lane_widths[-self._warmup:])
        avg_leftx = np.average(self._left_fitx[-6:], axis=0)
        avg_rightx = np.average(self._right_fitx[-6:], axis=0)
        for i in range(100):
            idx = np.arange(self._ploty.size, dtype=int)
            too_wide_ind = idx[rfitx - lfitx > lane_width + self._margin/10]
            too_narrow_ind = idx[rfitx - lfitx < lane_width - self._margin/10]
            if too_wide_ind.size == 0 and too_narrow_ind.size == 0:
                break
            # Fix where the width between the fitted lanes is
            # too wide or too narrow.
            for i in np.concatenate([too_wide_ind, too_narrow_ind]):
                #l_diff = abs(lfitx[i] - self._left_fitx[-1][i])
                #r_diff = abs(rfitx[i] - self._right_fitx[-1][i])
                l_diff = abs(lfitx[i] - avg_leftx[i])
                r_diff = abs(rfitx[i] - avg_rightx[i])
                # X-coordinate on the left deviated too much.
                if l_diff > r_diff:
                    # Set X-coordinate to be the midpoint between it
                    # and the previous frame.
                    lfitx[i] = (lfitx[i] + avg_leftx[i])/2
                else:
                    # Set X-coordinate to be the midpoint between it
                    # and the previous frame.
                    rfitx[i] = (rfitx[i] + avg_rightx[i])/2
        # Refit the lanes
        lfit = np.polyfit(self._ploty, lfitx, 2)
        rfit = np.polyfit(self._ploty, rfitx, 2)
        lfitx = lfit[0]*np.square(self._ploty) + \
                lfit[1]*self._ploty + lfit[2]
        rfitx = rfit[0]*np.square(self._ploty) + \
                rfit[1]*self._ploty + rfit[2]
        return lfit, rfit, lfitx, rfitx

    def _find_curvature(self, x):
        """ Calculate the lane curvatures in meters.

        Args:
            x: X-coordinates of the pixels in the lane.

        Returns
            The curvature of the lane.
        """
        y_eval = np.max(self._ploty)
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self._ploty * self._ym_per_pix,
                            x * self._xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*y_eval*self._ym_per_pix \
                          + fit_cr[1])**2) ** 1.5) / np.absolute(2 * fit_cr[0])
        return curverad

    def _fix_window(self, image, trim_top=0.25):
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
        left_lane_inds = ((nonzerox > (self._left_fit[0]*(nonzeroy**2) +
                                       self._left_fit[1]*nonzeroy +
                                       self._left_fit[2]-self._margin/5)) &
                          (nonzerox < (self._left_fit[0]*(nonzeroy**2) +
                                       self._left_fit[1]*nonzeroy +
                                       self._left_fit[2]+self._margin/5)))
        right_lane_inds = ((nonzerox > (self._right_fit[0]*(nonzeroy**2) +
                                        self._right_fit[1]*nonzeroy +
                                        self._right_fit[2]-self._margin/5)) &
                           (nonzerox < (self._right_fit[0]*(nonzeroy ** 2) +
                                        self._right_fit[1]*nonzeroy +
                                        self._right_fit[2]+self._margin/5)))
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