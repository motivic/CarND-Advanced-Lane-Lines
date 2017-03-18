#/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Johnson Jia"

import click
import cv2
import functools
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoClip, VideoFileClip
import numpy as np

from camera_calibration import Calibration
from edge_detection import abs_sobel_thresh, dir_thresh, mag_thresh
from lane_fitting import QuadraticLaneFitter

# Global constant calibrated using straight_lines1.jpg
# and straight_lines2.jpg
PERSPECTIVE_SRC_COORD = np.float32([[218.70, 710],
                                    [595.60, 450],
                                    [685.10, 450],
                                    [1087.30, 710]])

PERSPECTIVE_DST_COORD = np.float32([[300, 720],
                                    [300, 0],
                                    [1000, 0],
                                    [1000, 720]])

@click.command()
@click.option('--camera_cal', default='./camera_cal',
              help='folder containing the images for camera calibration')
@click.option('--nx', default=9, help='number of inside corners in x')
@click.option('--ny', default=6, help='number of inside corners in y')
@click.option('--img', default=0, help='1 if the MEDIA FILE is an image')
@click.option('--outfile', default='out.mp4', required=1)
@click.argument('media_file', required=1)
def main(camera_cal, nx, ny, img, outfile, media_file):
    """
    MEDIA FILE: The path to the video or image file to process.
    """
    # Calibrate the camera.
    calibration = Calibration(camera_cal, int(nx), int(ny))
    sample_chessboard = mpimg.imread('./camera_cal/calibration2.jpg')
    undistorted_cb = calibration.undistort(sample_chessboard)
    mpimg.imsave('output_images/undistort_chessboard.png', undistorted_cb)
    lane_fitter = QuadraticLaneFitter()

    if img != 1:
        process_frame = functools.partial(lane_detect,
                                          calibration=calibration,
                                          lane_fitter=lane_fitter,
                                          filename=media_file
                                            .split('/')[-1]
                                            .split('.')[0])
        clip = VideoFileClip(media_file)
        clip_w_lane = clip.fl_image(process_frame)
        clip_w_lane.write_videofile(outfile, audio=False)
    else:
        frame = mpimg.imread(media_file)
        laned_frame = lane_detect(frame,
                                  calibration=calibration,
                                  lane_fitter=lane_fitter,
                                  filename=media_file
                                    .split('/')[-1]
                                    .split('.')[0])
        mpimg.imsave(outfile, laned_frame)

def lane_detect(image, filename, calibration, lane_fitter):
    """ Apply camera calibration, edge detection, perspective
    transformation, and sliding window to detect lane lines.

    Args:
        image: The original image.
        filename: The name of the image file.
        calibration: Calibration object used to correct for camera warping.
        lane_fitter: QuadraticLaneFitter object used to fit the lanes.

    Returns:
        The undistorted image with lanes colored.
    """
    # Undistort the image.
    undistorted_img = calibration.undistort(image)
    '''plt.imshow(undistorted_img)
    plt.title('Undistorted version of {}.'.format(filename))
    plt.savefig('output_images/undistorted_{}.jpg'.format(filename))
    plt.plot([218.70, 595.60], [710, 450], 'r-')
    plt.plot([595.60, 685.10], [450, 450], 'r-')
    plt.plot([685.10, 1087.30], [450, 710], 'r-')
    plt.plot([218.70, 1087.30], [710, 710], 'r-')
    plt.title('Undistorted image with source points drawn.')
    plt.savefig('output_images/outlined_{}.jpg'.format(filename))'''

    # Perform gradient and saturation thresholding.
    img = np.copy(undistorted_img)
    binary = binary_filter(img)
    plt.imshow(binary, cmap='gray')
    plt.savefig('output_images/binary_{}.jpg'.format(filename))

    # Perform perspective transform.
    M = cv2.getPerspectiveTransform(PERSPECTIVE_SRC_COORD,
                                    PERSPECTIVE_DST_COORD)
    warped = cv2.warpPerspective(binary, M,
                                 binary.shape[0:2][::-1],
                                 flags=cv2.INTER_LINEAR)
    '''color_warped = cv2.warpPerspective(img, M,
                                       img.shape[0:2][::-1],
                                       flags=cv2.INTER_LINEAR)
    plt.imshow(color_warped)
    plt.title('Warped result with destination points drawn.')
    plt.plot([300, 1000], [720, 720], 'r-')
    plt.plot([300, 1000], [0, 0], 'r-')
    plt.plot([300, 300], [0, 720], 'r-')
    plt.plot([1000, 1000], [0, 720], 'r-')
    plt.savefig('output_images/warped_{}.jpg'.format(filename))
    plt.imshow(warped, cmap='gray')
    plt.show()
    '''

    # Fit quadratic polynomial to the lanes.
    out_img = lane_fitter.find_lanes(warped)

    # Draw lanes
    result =  draw_lane(warped,
                        lane_fitter.left_fitx,
                        lane_fitter.right_fitx,
                        lane_fitter.ploty,
                        M, undistorted_img)
    # Add curvature and offset from center information.
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "Radius of Curvature = {}m.\n" \
            .format((lane_fitter.left_curvature + lane_fitter.right_fitx)/2)
    if lane_fitter.offset < 0:
        title += "Vehicle is {}m left of center.".format(-lane_fitter.offset)
    else:
        title += "Vehicle is {}m right of center.".format(lane_fitter.offset)
    cv2.putText(result, title, (10, 10), font, 2, 255)
    return result

def draw_lane(warped, lx, rx, y, M, undist):
    """ Color the region between the lane lines.

    Args:
        warped: The warped image to color.
        lx: x coordinates of the left lane.
        rx: x coordinates of the right lane.
        y: y coordinates of either lane.
        M: The perspective transformation matrix.
        undist: The undistored image.

    Returns:
        The undistorted image with the lane colored.
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([lx, y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rx, y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using
    # inverse perspective matrix (Minv)
    Minv = np.linalg.inv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv,
                                  (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

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
    grad_filtered_binary[
                         (mag_binary == 1) &
                         (dir_binary == 1) &
                         (x_binary == 1) &
                         (y_binary == 1)
                        ] = 1
    # Apply filter on saturation channel
    saturation_filtered_binary = np.zeros_like(s_channel)
    saturation_filtered_binary[(s_channel > 100) &
                               (s_channel < 255)] = 1
    # Combine the two binary images
    filtered_binary = np.zeros_like(s_channel)
    filtered_binary[
                    (grad_filtered_binary == 1) |
                    (saturation_filtered_binary == 1)
                   ] = 1
    return filtered_binary

if __name__ == '__main__':
    main()