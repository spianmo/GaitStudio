from typing import Tuple, List


def calculateVelocity(point1: tuple, point2: tuple, time: float)->List[float]:
    velocity_x = (point2[0] - point1[0]) / time
    velocity_y = (point2[1] - point1[1]) / time
    velocity_z = (point2[2] - point1[2]) / time
    return velocity_x, velocity_y, velocity_z


def calculateAcceleration(point1: tuple, point2: tuple, time: float)->List[float]:
    acceleration = [velocity / time for velocity in calculateVelocity(point1, point2, time)]
    return acceleration


if __name__ == '__main__':
    result = calculateAcceleration(point1=(1, 1, 1), point2=(2, 5, 2), time=1.1)
    print(result)
