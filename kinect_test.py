import cv2
import numpy as np

import pyk4a
from kinect_helpers import colorize, color_depth_image, smooth_depth_image, depthInMeters
from pyk4a import Config, PyK4A


def main():
    smooth_depth_mode = False
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_1080P,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
            camera_fps=pyk4a.FPS.FPS_30,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    while True:
        capture = k4a.get_capture()
        if np.any(capture.depth):
            # 原始的RGBA视频帧
            color_image = capture.color
            # 深度图像数据
            depth_image = capture.transformed_depth
            # 归一化为米
            #depth_image = depthInMeters(depth_image)

            if smooth_depth_mode:
                transformed_colored_depth_image = color_depth_image(smooth_depth_image(depth_image, max_hole_size=10))
            else:
                transformed_colored_depth_image = color_depth_image(depth_image)

            # Combine both images
            combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, transformed_colored_depth_image, 0.3, 0)

            # Overlay body segmentation on depth image
            cv2.imshow('Transformed Color Depth Image', combined_image)

            cv2.imshow("k4a", colorize(capture.transformed_depth, (None, 5000), cv2.COLORMAP_HSV))

            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break
    k4a.stop()


if __name__ == "__main__":
    main()
