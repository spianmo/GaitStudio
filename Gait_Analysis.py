import cv2 as cv
import mediapipe as mp
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.signal import argrelextrema
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm


def auto_correlation(y, t) -> float:
    s = pd.Series(y)

    x = sm.tsa.acf(s)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.plot(t[:len(x)], x)
    plt.grid()
    plt.show()

    xf = np.copy(x)
    xf[0] = 0.0
    period = t[np.argmax(xf)]

    return period


def contact(y, fr, period, min_dis=9, max_dis=30) -> ndarray:
    # get the local minimums that are less than 150
    i_min = argrelextrema(y.values, np.less)[0]
    i_max = argrelextrema(y.values, np.greater)[0]

    i_s = []

    for i, _ in enumerate(i_min):
        for j, _ in enumerate(i_max):
            if i == 0:
                continue
            if (i_min[i] - i_min[i - 1]) > (i_max[j] - i_min[i - 1]) and (i_max[j] - i_min[i - 1]) > 0:
                if (y.iloc[i_max[j]] - y.iloc[i_min[i - 1]]) > 60:
                    i_s.append(i_min[i - 1])

    splits = y.iloc[i_s]

    # make sure we do not have two local minimums in less than 0.3 seconds.
    drop_index = []

    for loc, _ in enumerate(splits):
        if loc == 0:
            continue

        if (loc > 1):
            if (splits.index[loc - 1] in drop_index):
                if (splits.index[loc - 1] in drop_index) and (
                        ((splits.index[loc] - splits.index[loc - 2]) >= min_dis) and (
                        splits.index[loc] - splits.index[loc - 2]) <= max_dis):
                    continue

        if ((splits.index[loc] - splits.index[loc - 1]) < min_dis) or (
                splits.index[loc] - splits.index[loc - 1]) > max_dis:
            drop_index.append(splits.index[loc])

    # split the data in each cycle

    # print(splits)
    # print(drop_index)

    splits = splits.drop(drop_index).index

    t_diffs = []

    for i, s in enumerate(splits):

        if s == splits[0]:
            continue

        # y1_s = y1[splits[i-1]:s]

        # get the moment when the angle of the knee with the torso goes to greater than 180º
        # try:
        #    y1_s_180 = min(y1_s.index[y1_s>170].to_list())
        # except:
        #    continue

        # this is the moment where the foot do the first contact with a 0 angle between the knee and the torso

        # first_contact = y1_s_180 - 1

        # launching = y1_s.index[maximum(y1_s)[1]]

        y_s = y[splits[i - 1]:s]

        j_max = argrelextrema(y_s.values, np.greater)
        y_s_maxs = y_s.iloc[j_max]
        y_s_maxs = y_s_maxs[y_s_maxs.values > 135]
        # print(y2_s_maxs)
        # if y2_s_maxs.index[0] < 150:
        #    print(y2_s)
        #    print(y2_s_maxs)
        #    print(y2_s_maxs.index)

        first_contact = min(y_s_maxs.index)

        launching = max(y_s_maxs.index) + 1

        # print("First contact: ", first_contact)
        # print("Launching: ", launching)

        # get the moment when the knee angle is in its minimum
        # launching  = y2_s.index[minimum(y2_s)[1]]

        t = (1.0 / fr) * (launching - first_contact)

        if (t > 0.1) & (t < 2 * period / 3):
            t_diffs.append(t)

    t_diff = np.mean(t_diffs)

    return t_diff


def analysis(df_angles: DataFrame, fps: int):
    period = auto_correlation(df_angles["RHip_angle"], df_angles["Time_in_sec"])
    print("每个步骤之间的时间段:", period / 2, "秒")
    print("步速:", (2 * 60.0) / period, "步数/分钟")

    contact_time_r = contact(df_angles["RKnee_angle"], fps, period, min_dis=int(period * fps) - 5,
                             max_dis=int(period * fps) + 5)
    print("右脚触地时间:", contact_time_r, "s")
    contact_time_l = contact(df_angles["LKnee_angle"], fps, period, min_dis=int(period * fps) - 5,
                             max_dis=int(period * fps) + 5)
    print("左脚触地时间:", contact_time_l, "s")

    flight_ratio_r = 100 * (1 - (2 * contact_time_r / period))
    print("右脚空中时间比:", flight_ratio_r, "%")
    flight_ratio_l = 100 * (1 - (2 * contact_time_l / period))
    print("左脚空中时间比:", flight_ratio_l, "%")
