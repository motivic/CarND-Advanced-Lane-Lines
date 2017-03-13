# -*- coding: utf-8 -*-

__author__ = 'Johnson Jia'

import cv2
import numpy as np

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """ Apply threshold on Sobel gradients along a single direction.

    Args:
        img: The numpy array containing the image.
        orient: The orientation, can be either 'x' or 'y'.
        sobel_kernel: The Sobel kernel.
        thresh: The thresholds (lower, upper) to apply to the result.

    Returns:
        The filtered binary image as a numpy array.
    """
    sobel = cv2.Sobel(img, cv2.CV_64F, orient == 'x',
                      orient == 'y', ksize=sobel_kernel)
    sobel_abs = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * sobel_abs / np.max(sobel_abs))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return grad_binary

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    """ Apply thresholds on the magnitude of the Sobel gradients.

    Args:
        img: The numpy array containing the image.
        sobel_kernel: The Sobel kernel size.
        thresh: The thresholds (lower and upper) to apply to the result.

    Returns:
        The filtered binary image as a numpy array.
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelxy = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*sobelxy/np.max(sobelxy))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return mag_binary

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """ Apply thresholds on the direction (angle) of the Sobel gradients.

    Args:
        img: The numpy array containing the image.
        sobel_kernel: The Sobel kernel.
        thresh: The thresholds (lower and upper) to apply to the result.

    Returns:
        The filtered binary image as a numpy array.
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    direction = np.arctan2(sobely, sobelx)
    direction = np.absolute(direction)
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return dir_binary