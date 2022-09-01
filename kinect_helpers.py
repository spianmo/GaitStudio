from typing import Optional, Tuple

import cv2
import numpy as np

from pyk4a import ImageFormat


def color_depth_image(depth_image):
    depth_color_image = cv2.convertScaleAbs(depth_image, alpha=0.05)  # alpha is fitted by visual comparison with Azure k4aviewer results
    depth_color_image = cv2.applyColorMap(depth_color_image, cv2.COLORMAP_JET)
    return depth_color_image


def smooth_depth_image(depth_image, max_hole_size=10):
    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    mask[depth_image == 0] = 1

    kernel = np.ones((max_hole_size, max_hole_size), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    mask = mask - erosion

    smoothed_depth_image = cv2.inpaint(depth_image.astype(np.uint16), mask, max_hole_size, cv2.INPAINT_NS)
    return smoothed_depth_image


def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    # examples for all possible pyk4a.ColorFormats
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
        # this also works and it explains how the COLOR_NV12 color color_format is stored in memory
        # h, w = color_image.shape[0:2]
        # h = h // 3 * 2
        # luminance = color_image[:h]
        # chroma = color_image[h:, :w//2]
        # color_image = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image


def colorize(
        image: np.ndarray,
        clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
        colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


def depthInMeters(rawDepthArray):
    return rawDepthArray / 1000
