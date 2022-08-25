import os
import time
from io import BytesIO
from math import inf
from pathlib import Path
from typing import List

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

from reports.GenarateGaitReport import HealBoneGaitReport

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def auto_correlation(y, t) -> float:
    """
    自相关函数
    :param y:
    :param t:
    :return:
    """
    s = pd.Series(y)

    x = sm.tsa.acf(s)

    # plt.plot(t[:len(x)], x)
    # plt.grid()
    # plt.show()

    xf = np.copy(x)
    xf[0] = 0.0
    period = t[xf.size]
    return period


def get_local_format_time(timestamp):
    local_time = time.localtime()
    format_time = time.strftime("%Y%m%d%H%M%S", local_time)
    return format_time


def analysis(df_angles: DataFrame, fps: int):
    period = auto_correlation(df_angles["RHip_angle"], df_angles["Time_in_sec"])
    print("样本步行总时长: ", df_angles["Time_in_sec"].iloc[-1], "秒")
    print("每步之间的平均时长:", period / 2, "秒")
    print("步速:", ((2 * 60.0) / period).round(2), "步数/分钟")

    print("最大躯干与胯骨的夹角:", df_angles["TorsoLHip_angle"].max().round(2), "°")
    print("最小躯干与胯骨的夹角:", df_angles["TorsoLHip_angle"].min().round(2), "°")

    print("最大左股骨与胯骨的夹角:", df_angles["LHip_angle"].max().round(2), "°")
    print("最大左股骨与胯骨的夹角:", df_angles["LHip_angle"].min().round(2), "°")

    print("最大右股骨与胯骨的夹角:", df_angles["RHip_angle"].max().round(2), "°")
    print("最小右股骨与胯骨的夹角:", df_angles["RHip_angle"].min().round(2), "°")

    print("最大左胫骨与左股骨的夹角:", df_angles["LKnee_angle"].max().round(2), "°")
    print("最小左胫骨与左股骨的夹角:", df_angles["LKnee_angle"].min().round(2), "°")

    print("最大右胫骨与右股骨的夹角:", df_angles["RKnee_angle"].max().round(2), "°")
    print("最小右胫骨与右股骨的夹角:", df_angles["RKnee_angle"].min().round(2), "°")

    print("最大左踝与左胫骨的夹角:", df_angles["LAnkle_angle"].max().round(2), "°")
    print("最小左踝与左胫骨的夹角:", df_angles["LAnkle_angle"].min().round(2), "°")

    print("最大右踝与右胫骨的夹角:", df_angles["RAnkle_angle"].max().round(2), "°")
    print("最小右踝与右胫骨的夹角:", df_angles["RAnkle_angle"].min().round(2), "°")

    report_output = Path("report_output")
    if not report_output.is_dir():
        os.makedirs(report_output)

    report = HealBoneGaitReport('report_output/GaitReport-' + get_local_format_time(time.time()) + '.pdf', summaryData=[
        ["检测项", "检测值", "单位", "参考值"],
        ["样本步行总时长", str(df_angles["Time_in_sec"].iloc[-1]), "秒", "-"],
        ["每步之间的平均时长", str(period / 2), "秒", "-"],
        ["步速:", str(((2 * 60.0) / period).round(2)), "步数/分钟", "-"],

        ["最大躯干与胯骨的夹角", str(df_angles["TorsoLHip_angle"].max().round(2)), "°", "-"],
        ["最小躯干与胯骨的夹角", str(df_angles["TorsoLHip_angle"].min().round(2)), "°", "-"],

        ["最大左股骨与胯骨的夹角", str(df_angles["LHip_angle"].max().round(2)), "°", "-"],
        ["最大左股骨与胯骨的夹角", str(df_angles["LHip_angle"].min().round(2)), "°", "-"],

        ["最大右股骨与胯骨的夹角", str(df_angles["RHip_angle"].max().round(2)), "°", "-"],
        ["最小右股骨与胯骨的夹角", str(df_angles["RHip_angle"].min().round(2)), "°", "-"],

        ["最大左胫骨与左股骨的夹角", str(df_angles["LKnee_angle"].max().round(2)), "°", "-"],
        ["最小左胫骨与左股骨的夹角", str(df_angles["LKnee_angle"].min().round(2)), "°", "-"],

        ["最大右胫骨与右股骨的夹角", str(df_angles["RKnee_angle"].max().round(2)), "°", "-"],
        ["最小右胫骨与右股骨的夹角", str(df_angles["RKnee_angle"].min().round(2)), "°", "-"],

        ["最大左踝与左胫骨的夹角", str(df_angles["LAnkle_angle"].max().round(2)), "°", "-"],
        ["最小左踝与左胫骨的夹角", str(df_angles["LAnkle_angle"].min().round(2)), "°", "-"],

        ["最大右踝与右胫骨的夹角", str(df_angles["RAnkle_angle"].max().round(2)), "°", "-"],
        ["最小右踝与右胫骨的夹角", str(df_angles["RAnkle_angle"].min().round(2)), "°", "-"],
        ["检测项共计", "", "", "17 项"]
    ], graph=df_angles)
    report.exportPDF()
    df_angles.to_excel("report_output/GaitAngle-" + get_local_format_time(time.time()) + ".xlsx")
