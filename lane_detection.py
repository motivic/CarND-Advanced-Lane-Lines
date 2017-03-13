#/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Johnson Jia"

import click
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from camera_calibration import Calibration
from edge_detection import abs_sobel_thresh, dir_thresh, mag_thresh
from lane_fitting import Quadratic_lane_fitter

# Global constant calibrated using straight_lines1.jpg
# and straight_lines2.jpg
PERSPECTIVE_SRC_COORD = np.float32([[218.44, 710],
                                    [533.23, 480],
                                    [729.81, 480],
                                    [1086.35, 710]])

PERSPECTIVE_DST_COORD = np.float32([[300, 720],
                                    [300, 100],
                                    [950, 100],
                                    [950, 720]])

@click.command()
@click.option('--camera_cal',
              help='folder containing the images for camera calibration')
@click.option('--nx', default=9, help='number of inside corners in x')
@click.option('--ny', default=6, help='number of inside corners in y')
@click.argument('image', required=1)
def detect(camera_cal, nx, ny, image):
    """ IMAGE: Image to perform lane detection
    """
    # Calibrate the camera.
    calibration = Calibration(camera_cal, int(nx), int(ny))
    img = mpimg.imread(image)

    # Undistort the image.
    undistorted_img = calibration.undistort(img)
    plt.imshow(undistorted_img)
    plt.plot([218.44, 533.23], [710, 480], 'r-')
    plt.plot([729.81, 1086.35], [480, 710], 'r-')
    plt.show()
    cv2.imwrite('undistorted.jpg', undistorted_img)

    # Perform edge detection.
    img = np.copy(undistorted_img)
    binary = binary_filter(img)
    plt.imshow(binary, cmap='gray')
    plt.show()

    # Perform perspective transform.
    M = cv2.getPerspectiveTransform(PERSPECTIVE_SRC_COORD,
                                    PERSPECTIVE_DST_COORD)
    warped = cv2.warpPerspective(binary, M,
                                 binary.shape[0:2][::-1],
                                 flags=cv2.INTER_LINEAR)
    plt.imshow(warped, cmap='gray')
    plt.show()

    # Fit quadratic polynomial to the lanes.
    lane_fitting = Quadratic_lane_fitter()
    lane_fitting.find_lanes(warped, True)

def binary_filter(img):
    """ Apply Sobel filters and saturation filters.

    Args:
        img: The original image to apply thresholds to.

    Returns:
        A black-and-white filtered image.
    """
    # Convert to HLS color space.
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Apply gradient/Sobel filtering
    mag_binary = mag_thresh(l_channel, thresh=(30, 200))
    dir_binary = dir_thresh(l_channel, thresh=(np.pi/4-0.8, np.pi/4+0.8))
    x_binary = abs_sobel_thresh(l_channel, orient='x', thresh=(15, 200))
    y_binary = abs_sobel_thresh(l_channel, orient='y', thresh=(15, 200))
    grad_filtered_binary = np.zeros_like(l_channel)
    grad_filtered_binary[(mag_binary == 1) &
                         (dir_binary == 1) &
                         (x_binary == 1) &
                         (y_binary == 1)] = 1
    # Apply filter on saturation channel
    saturation_filtered_binary = np.zeros_like(s_channel)
    saturation_filtered_binary[(s_channel > 90) &
                               (s_channel < 255)] = 1
    # Combine the two binary images
    filtered_binary = np.zeros_like(s_channel)
    filtered_binary[(grad_filtered_binary == 1) |
                    (saturation_filtered_binary == 1)] = 1
    return filtered_binary

def perspective_transform(img, ):
    pass

if __name__ == '__main__':
    detect()