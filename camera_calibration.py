# -*- coding: utf-8 -*-

__author__ = 'Johnson Jia'

import cv2
import glob
import numpy as np


class Calibration():
    """ Perform camera calibration and Keep track of camera calibration results.

    Attributes:
        chessboard_with_corners: A list of the chessboard images used in
            calibration with corners drawn.
    """

    def __init__(self, img_folder, nx, ny):
        """ Initializes and set calibration parameters.

        Args:
            img_folder: Folder containing the calibration chessboard images.
            nx: The number of insider corners in the horizontal direction.
            ny: The number of insider corners in the vertical direction.
        """
        self._img_folder = img_folder
        self._nx = nx
        self._ny = ny
        self._find_corners()

    def _find_corners(self):
        """ Find corners following the directions available here:
                https://goo.gl/RmJsxW.

            Set `objpoints` and `imgpoints` used in calibrating camera.
        """
        # Termination criteria.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(5,8,0).
        objp = np.zeros((self._ny * self._nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self._ny, 0:self._nx].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self._objpoints = []  # 3D point in real world space
        self._imgpoints = []  # 2D points in image plane
        self._img_with_corners = []  # Original images with corners drawn

        images = glob.glob(self._img_folder + '/*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners.
            ret, corners = cv2.findChessboardCorners(gray, (self._ny, self._nx), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                self._objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self._imgpoints.append(corners2)

                # Draw the corners and save images in `img_with_corners`
                self._img_with_corners.append(
                    cv2.drawChessboardCorners(img,
                                              (self._ny, self._nx),
                                              corners2,
                                              ret)
                )

    def chessboard_with_corners(self):
        """ Iterator that yields chessboards with corners.

        Yields:
            Chessboard images with corners drawn.
        """
        for img in self._img_with_corners:
            yield img

    def _calibrate_camera(self, image):
        """ Set camera calibration parameters.

        Args:
            image: The image that we calibrate against.
        """
        image_size = (image.shape[1], image.shape[0])
        _, self._mtx, self._dist, _, _ = cv2.calibrateCamera(self._objpoints,
                                                             self._imgpoints,
                                                             image_size,
                                                             None, None)

    def undistort(self, image):
        """ Undistort the image.

        Args:
            image: The image to undistort.

        Returns:
            The undistorted image.
        """
        self._calibrate_camera(image)
        return cv2.undistort(image, self._mtx, self._dist, None, self._mtx)
