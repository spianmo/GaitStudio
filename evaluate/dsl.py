from __future__ import division

from typing import Dict

import numpy as np
import pyparsing as pyp
import math
import operator

from mediapipe.python.solutions.pose import PoseLandmark
from numpy import ndarray


def vector(keyPoints, keyPointIndex: PoseLandmark) -> ndarray:
    return np.array(
        [keyPoints[keyPointIndex][0], keyPoints[keyPointIndex][1], keyPoints[keyPointIndex][2]])


def angle(vector1, vector2, m=False) -> float:
    x = np.dot(vector1, -vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    theta = np.degrees(np.arccos(x))
    return theta if not m else 180 - theta


def lx(vector):
    return np.array(0, vector[1], vector[2])


def ly(vector):
    return np.array(vector[0], 0, vector[1])


def lz(vector):
    return np.array(vector[0], vector[1], 0)


def DSL(expStr: str, vars: dict):
    return eval(expStr.format(**vars))


if __name__ == '__main__':
    bones = {
        "$bone1": 11
    }
    # print(DSL('{$bone1} in range(0, 20)', bones))
    print(angle(np.array([-0.2, 1, 0]), np.array([0, 1, 0]), m=True))
