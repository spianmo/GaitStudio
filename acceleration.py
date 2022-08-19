from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray


def calculateVelocity(point1: list, point2: list, time: float) -> Tuple[float, float, float]:
    velocity_x = (point2[0] - point1[0]) / time
    velocity_y = (point2[1] - point1[1]) / time
    velocity_z = (point2[2] - point1[2]) / time
    return velocity_x, velocity_y, velocity_z


def calculateAcceleration(point1: list, point2: list, time: float) -> List[float]:
    acceleration = [velocity / time for velocity in calculateVelocity(point1, point2, time)]
    return acceleration


def calculateAccelerationList(point_list: list, time: float) -> List[List]:
    acceleration_list = []
    for i in range(len(point_list) - 1):
        acceleration_list.append(calculateAcceleration(point_list[i], point_list[i + 1], time))
    return acceleration_list


def calculateAccelerationListFrame(point_list: list, frames: int) -> Tuple[ndarray, ndarray, ndarray]:
    accelerations = calculateAccelerationList(point_list, 1 / frames)
    return np.array([acceleration[0] for acceleration in accelerations]), np.array(
        [acceleration[1] for acceleration in accelerations]), np.array(
        [acceleration[2] for acceleration in accelerations])


def sensormotionDemo(pts_cam: ndarray, fps: int):
    accelerations_x, accelerations_y, accelerations_z = calculateAccelerationListFrame(pts_cam.tolist(), fps)

    frames_time = np.array([frame_index * 1000 / fps for frame_index in range(len(pts_cam) - 1)])

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))

    ax[0].set_title('Medio-lateral (ML) - side to side')
    ax[0].plot(frames_time, accelerations_x, linewidth=0.5, color='k')

    ax[1].set_title('Vertical (VT) - up down')
    ax[1].plot(frames_time, accelerations_y, linewidth=0.5, color='k')

    ax[2].set_title('Antero-posterior (AP) - forwards backwards')
    ax[2].plot(frames_time, accelerations_z, linewidth=0.5, color='k')

    fig.subplots_adjust(hspace=.5)
