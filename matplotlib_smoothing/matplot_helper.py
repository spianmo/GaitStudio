import numpy as np


def moving_average(interval, windowSize=10):
    window = np.ones(int(windowSize)) / float(windowSize)
    re = np.convolve(interval, window, 'same')
    return re
