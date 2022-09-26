from __future__ import division

from typing import Dict

import numpy as np
import pyparsing as pyp
import math
import operator
import time

from mediapipe.python.solutions.pose import PoseLandmark
from numpy import ndarray


def vector(keyPoints, keyPointIndex: PoseLandmark) -> ndarray:
    return np.array(
        [keyPoints[keyPointIndex][0], keyPoints[keyPointIndex][1], keyPoints[keyPointIndex][2]])


def angle(vector1, vector2, m=False) -> float:
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    x = np.dot(vector1, -vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    theta = np.degrees(np.arccos(x))
    return theta if not m else 180 - theta


def lx(vector):
    return [0, vector[1], vector[2]]


def ly(vector):
    return [vector[0], 0, vector[1]]


def lz(vector):
    return [vector[0], vector[1], 0]


def credible_pose(keypoints, credit=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)):
    """
    检测Pose可信度
    :param credit:
    :param keypoints:
    :return:
    """
    confidences = [keypoint[3] if index in credit else 1 for index, keypoint in enumerate(keypoints)]
    return np.array(confidences).min()


def currentTime():
    return time.time()

def _T(npVec):
    return str(list(npVec))

def DSL(expStr: str, vars: dict):
    return eval(expStr.format(**vars))


if __name__ == '__main__':
    bones = {
        "$bone1": 11,
        "v1": np.array([-0.2, 1, 0]),
        "v2": np.array([0, 1, 0])
    }
    print(_T(bones["v1"]))
    # print(DSL('{$bone1} in range(0, 20)', bones))

    print(DSL("angle({_T(v1)}, {_T(v2)}, m=True)", bones))
    # print(DSL("time.time()", {}))
    # xx = (angle(ly(torso), torso) in range(0, 50)) and angle(ly(femur), femur) > 30 and angle(ly(tibia), tibia) > 30
