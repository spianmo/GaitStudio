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

positions_2d = {}
positions_3d = {}
angles_3d = {}


def simple_plot_angles(title: str, df: DataFrame) -> None:
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
    video_file = Path('../data/multi-hb/Walking.camera2.mp4')

    cam = video_file.stem.split('.')[1]
    video = Video(video_file)

    # estimate 2D and 3D keypoints using the HPE pipeline
    estimator_2d = MediaPipe_Estimator2D(out_format='coco')
    estimator_3d = VideoPose3D()

    kpts, meta = estimator_2d.estimate(video)
    pose_3d = estimator_3d.estimate(kpts, meta)['video']
    angles = calc_common_angles(pose_3d)
    angles['Frames'] = [i + 1 for i in range(len(angles['RKnee']))]

    # save data at correct list position
    pose_2d = kpts['video']['custom'][0]
    positions_2d = pose_2d
    positions_3d = pose_3d
    angles_3d = angles
    tmp = pd.DataFrame(angles)
    simple_plot_angles('Test', tmp)

    print(positions_3d)
    print('\n\n')
