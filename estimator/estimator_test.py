from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from estimator.mediapipe_estimator import MediaPipe_Estimator2D
from estimator.utils.angle_helper import calc_common_angles
from estimator.utils.data_utils import suggest_metadata
from estimator.video import Video
from estimator.videopose3d import VideoPose3D
import seaborn as sns
from matplotlib import pyplot as plt

# map cam id to list index (from VideoPose3D)
cam_map = {
    'camera1': 0,
    'camera2': 1,
}

positions_2d = {}
positions_3d = {}
angles_3d = {}

in_dir = Path('../data/multi-virtual')


def plot_angles(title: str, df: DataFrame) -> None:
    rc = {'font.sans-serif': 'SimHei',
          'axes.unicode_minus': False}

    sns.set_style(style='darkgrid', rc=rc)

    fig, axes = plt.subplots(2, 2, figsize=(24, 14))

    fig.suptitle("关节角度变化周期 - " + title)

    sns.lineplot(ax=axes[0, 0], data=df, x="Frames", y="RKnee").set(xlabel="帧数",
                                                                    ylabel="RKnee (°)")
    sns.lineplot(ax=axes[1, 0], data=df, x="Frames", y="LKnee").set(xlabel="帧数",
                                                                    ylabel="LKnee (°)")

    plt.show()


if __name__ == '__main__':

    # Iterate over all activities
    positions_2d = [None] * len(cam_map)
    positions_3d = [None] * len(cam_map)
    angles_3d = [None] * len(cam_map)

    # every file is by one camera
    for video_files in in_dir.iterdir():
        cam = video_files.stem.split('.')[1]
        video = Video(video_files)

        # estimate 2D and 3D keypoints using the HPE pipeline
        estimator_2d = MediaPipe_Estimator2D(out_format='coco')
        estimator_3d = VideoPose3D()

        kpts, meta = estimator_2d.estimate(video)
        pose_3d = estimator_3d.estimate(kpts, meta)['video']
        angles = calc_common_angles(pose_3d)
        angles['Frames'] = [i+1 for i in range(len(angles['RKnee']))]

        # save data at correct list position
        id = cam_map[cam]
        pose_2d = kpts['video']['custom'][0]
        positions_2d[id] = pose_2d
        positions_3d[id] = pose_3d
        angles_3d[id] = angles
        tmp = pd.DataFrame(angles)
        plot_angles('Test', tmp)

    print(positions_3d)
    print('\n\n')
