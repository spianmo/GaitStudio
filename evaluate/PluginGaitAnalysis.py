import os
import time
from io import BytesIO
from math import inf
from pathlib import Path
from typing import List, Tuple
import numpy as np
from numpy import ndarray
from pandas import DataFrame

from matplotlib import pyplot as plt
import sensormotion as sm
import statsmodels.api as smapi
from qtmodernredux import QtModernRedux
from svglib.svglib import svg2rlg


plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

plt.ioff()


def polt_accelerations(frames_time, x, y, z, x_f, y_f, z_f):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5.5, 7))

    fig.suptitle("RIGHT_KNEE Accelerations 右膝关节-加速度周期分解")

    ax[0].set_title('Medio-lateral (ML) - side to side 加速度X轴')
    ax[0].plot(frames_time, x, linewidth=0.3, color='k')
    ax[0].plot(frames_time, x_f, linewidth=0.8, color='r')
    ax[0].set_xlabel("时间（秒）")
    ax[0].set_ylabel("米/秒$^{2}$")

    ax[1].set_title('Vertical (VT) - up down 加速度Y轴')
    ax[1].plot(frames_time, y, linewidth=0.3, color='k')
    ax[1].plot(frames_time, y_f, linewidth=0.9, color='r')
    ax[1].set_xlabel("时间（秒）")
    ax[1].set_ylabel("米/秒$^{2}$")

    ax[2].set_title('Antero-posterior (AP) - forwards backwards 加速度Z轴')
    ax[2].plot(frames_time, z, linewidth=0.3, color='k')
    ax[2].plot(frames_time, z_f, linewidth=0.9, color='r')
    ax[2].set_xlabel("时间（秒）")
    ax[2].set_ylabel("米/秒$^{2}$")

    fig.tight_layout()
    image = BytesIO()
    fig.savefig(image, format='svg')
    image.seek(0)
    return svg2rlg(image)


def polt_find_peaks(time, signal, peak_type="peak", min_val=0.5, min_dist=25, detrend=0, show_grid=True,
                    fig_size=(5.5, 3.5)):
    time = np.array(time)
    signal = np.array(signal)

    # Check for detrend
    if detrend == 0:  # No detrending - don't calculate baseline
        new_signal = signal
    else:  # Detrend the signal
        new_signal = sm.signal.detrend_signal(signal, detrend)

    # Check peak type
    if peak_type == "peak":
        # Original input signal
        peaks = sm.peak.indexes(new_signal, thres=min_val, min_dist=min_dist)
    elif peak_type == "valley":
        # Flip the input signal for valleys
        peaks = sm.peak.indexes(np.negative(new_signal), thres=min_val, min_dist=min_dist)
    elif peak_type == "both":
        peaks = sm.peak.indexes(new_signal, thres=min_val, min_dist=min_dist)
        valleys = sm.peak.indexes(np.negative(new_signal), thres=min_val, min_dist=min_dist)
        peaks = np.sort(np.append(peaks, valleys))

    if detrend == 0:
        f, axarr = plt.subplots(1, 1, figsize=fig_size)
        axarr.plot(time, signal, "k")
        axarr.plot(
            time[peaks],
            signal[peaks],
            "r+",
            ms=15,
            mew=2,
            label="{} peaks".format(len(peaks)),
        )
        axarr.set_xlim(min(time), max(time))
        axarr.set_xlabel("时间（秒）")
        axarr.grid(show_grid)
        axarr.set_ylabel("米/秒$^{2}$")
        axarr.legend(loc="lower right")
    else:
        f, axarr = plt.subplots(2, 1, figsize=fig_size)
        axarr[0].plot(time, signal, "k")
        axarr[0].title.set_text("Original")
        axarr[0].set_xlim(min(time), max(time))
        axarr[0].set_xlabel("时间（秒）")
        axarr[0].set_ylabel("米/秒$^{2}$")
        axarr[0].grid(show_grid)

        axarr[1].plot(time, new_signal, "k")
        axarr[1].plot(
            time[peaks],
            new_signal[peaks],
            "r+",
            ms=15,
            mew=2,
            label="{} peaks".format(len(peaks)),
        )
        axarr[1].title.set_text("Detrended (degree: {})".format(detrend))
        axarr[1].set_xlim(min(time), max(time))
        axarr[1].set_xlabel("时间（秒）")
        axarr[1].set_ylabel("米/秒$^{2}$")
        axarr[1].grid(show_grid)
        axarr[1].legend(loc="lower right")

    f.tight_layout()
    if peak_type == "peak":
        suptitle_string = "Peak Detection (val: {}, dist: {}) 峰值信号"
    else:
        suptitle_string = "Valley Detection (val: {}, dist: {}) 谷值信号"
    plt.suptitle(suptitle_string.format(min_val, min_dist), y=1.01)
    image = BytesIO()
    f.savefig(image, format='svg')
    image.seek(0)

    if detrend == 0:
        return time[peaks], signal[peaks], svg2rlg(image)
    else:
        return time[peaks], new_signal[peaks], new_signal, svg2rlg(image)


def plot_signal(
        time,
        signal,
        title="",
        xlab="",
        ylab="",
        line_width=1,
        alpha=1,
        color="k",
        subplots=False,
        show_grid=True,
        fig_size=(10, 5),
):
    if type(signal) == list:  # Multiple lines to be plotted
        if subplots:
            f, axarr = plt.subplots(len(signal), 1, figsize=fig_size)
        else:
            f, axarr = plt.subplots(figsize=fig_size)

        for i, line in enumerate(signal):  # Iterate through each plot line
            cur_data = line["data"]

            # Get options for the current line
            try:
                cur_label = line["label"]
            except KeyError:
                print("Warning: Label missing for signal")
                cur_label = ""
            try:
                cur_color = line["color"]
            except KeyError:
                cur_color = color
            try:
                cur_alpha = line["alpha"]
            except KeyError:
                cur_alpha = alpha
            try:
                cur_linewidth = line["line_width"]
            except KeyError:
                cur_linewidth = line_width

            if subplots:  # Show lines in separate plots, in the same figure
                axarr[i].plot(
                    time,
                    cur_data,
                    label=cur_label,
                    linewidth=cur_linewidth,
                    alpha=cur_alpha,
                    color=cur_color,
                )

                axarr[i].set_xlim(min(time), max(time))
                axarr[i].set_xlabel(xlab)
                axarr[i].set_ylabel(ylab)
                axarr[i].grid(show_grid)
                axarr[i].legend()
                f.subplots_adjust(hspace=0.5)
            else:  # Show all lines on the same plot
                axarr.plot(
                    time,
                    cur_data,
                    label=cur_label,
                    linewidth=cur_linewidth,
                    alpha=cur_alpha,
                    color=cur_color,
                )

                axarr.set_xlim(min(time), max(time))
                axarr.set_xlabel(xlab)
                axarr.set_ylabel(ylab)
                axarr.grid(show_grid)
                axarr.legend()
    else:  # Single line plot
        f, axarr = plt.subplots(figsize=fig_size)
        axarr.plot(time, signal, linewidth=line_width, alpha=alpha, color=color)
        axarr.set_xlim(min(time), max(time))
        axarr.set_xlabel(xlab)
        axarr.set_ylabel(ylab)
        axarr.grid(show_grid)

    plt.suptitle(title)
    image = BytesIO()
    f.tight_layout()
    f.savefig(image, format='svg')
    image.seek(0)
    return svg2rlg(image)


def plot_xcorr(x, y, scale="none", show_grid=True, fig_size=(5.5, 3.5)):
    x = np.array(x)
    y = np.array(y)

    # Pad shorter array if signals are different lengths
    if x.size > y.size:
        pad_amount = x.size - y.size
        y = np.append(y, np.repeat(0, pad_amount))
    elif y.size > x.size:
        pad_amount = y.size - x.size
        x = np.append(x, np.repeat(0, pad_amount))

    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-(x.size - 1), x.size)

    # Scale the correlation values
    # Equivalent to xcorr scaling options in MATLAB
    if scale == "biased":
        corr = corr / x.size
    elif scale == "unbiased":
        corr /= x.size - abs(lags)
    elif scale == "coeff":
        corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))
    drawing = plot_signal(
        lags,
        corr,
        title="Cross-correlation (scale: {}) 互相关函数(无偏估计)".format(scale),
        xlab="Lag 延迟",
        ylab="Correlation 相关性系数",
        show_grid=show_grid,
        fig_size=fig_size,
    )
    return corr, lags, drawing


def plot_cut_points(x, set_name, n_axis, fig_size=(5.5, 3.5)):
    sets = {
        "butte_preschoolers": {
            1: {
                "sedentary": [-np.inf, 239],
                "light": [240, 2119],
                "moderate": [2120, 4449],
                "vigorous": [4450, np.inf],
            },
            3: {
                "sedentary": [-np.inf, 819],
                "light": [820, 3907],
                "moderate": [3908, 6111],
                "vigorous": [6112, np.inf],
            },
        },
        "freedson_adult": {
            1: {
                "sedentary": [-np.inf, 99],
                "light": [100, 1951],
                "moderate": [1952, 5724],
                "vigorous": [5725, 9498],
                "very vigorous": [9499, np.inf],
            },
            3: {
                "light": [-np.inf, 2690],
                "moderate": [2691, 6166],
                "vigorous": [6167, 9642],
                "very vigorous": [9643, np.inf],
            },
        },
        "freedson_children": {
            1: {
                "sedentary": [-np.inf, 149],
                "light": [150, 499],
                "moderate": [500, 3999],
                "vigorous": [4000, 7599],
                "very vigorous": [7600, np.inf],
            }
        },
        "keadle_women": {
            1: {
                "sedentary": [-np.inf, 99],
                "light": [100, 1951],
                "moderate": [1952, np.inf],
            },
            3: {
                "sedentary": [-np.inf, 199],
                "light": [200, 2689],
                "moderate": [2690, np.inf],
            },
        },
    }

    try:
        cur_set = sets[set_name][n_axis]
        print("Cut-point set: {} (axis count: {})...".format(set_name, n_axis))

        for i in cur_set:
            print("{}: {} to {}".format(i, cur_set[i][0], cur_set[i][1]))
    except KeyError:
        print(
            "Error: cut-point set not found. Make sure the set name and/or "
            "number of axes are correct"
        )
        raise

    # categorize counts
    category = []
    for count in x:
        for intensity in cur_set:
            if cur_set[intensity][0] <= count <= cur_set[intensity][1]:
                category.append(intensity)
                break

    # count time spent
    category_unique, category_count = np.unique(category, return_counts=True)
    time_spent = np.asarray((category_unique, category_count))

    # plot counts with intensity categories
    boundaries = [(item, cur_set[item][0]) for item in cur_set]
    boundaries.sort(key=lambda x: x[1])

    f, ax = plt.subplots(1, 1, figsize=fig_size)

    ax.bar(range(1, len(x) + 1), x)

    for line in boundaries[1:]:
        if line[1] < max(x):
            plt.axhline(line[1], linewidth=1, linestyle="--", color="k")
            t = plt.text(0.4, line[1], line[0], backgroundcolor="w")
            t.set_bbox(dict(facecolor="w", edgecolor="k"))

    plt.xticks(range(1, len(x) + 1))

    plt.suptitle("Physical activity counts and intensity 物理活动次数和强度")
    plt.xlabel("Epoch (length: 60 seconds)")
    plt.ylabel("PA count")
    image = BytesIO()
    f.savefig(image, format='svg')
    image.seek(0)
    return category, time_spent, svg2rlg(image)


def calculateVelocity(point1: list, point2: list, time: float) -> Tuple[float, float, float]:
    velocity_x = (point2[0] - point1[0]) / time
    velocity_y = (point2[1] - point1[1]) / time
    velocity_z = (point2[2] - point1[2]) / time
    return velocity_x, velocity_y, velocity_z


def calculateAcceleration(point1: list, point2: list, time: float) -> List[float]:
    acceleration = [velocity / time for velocity in calculateVelocity(point1, point2, time)]
    return acceleration


def calculateAccelerationList(point_list: list, frames_time_sec_raw) -> List[List]:
    acceleration_list = []
    for i in range(len(point_list) - 1):
        acceleration_list.append(
            calculateAcceleration(point_list[i], point_list[i + 1],
                                  frames_time_sec_raw[i + 1] - frames_time_sec_raw[i]))
    return acceleration_list


def calculateAccelerationListFrame(point_list: list, frames_time_sec_raw) -> Tuple[ndarray, ndarray, ndarray]:
    accelerations = calculateAccelerationList(point_list, frames_time_sec_raw)
    return np.array([acceleration[0] for acceleration in accelerations]), np.array(
        [acceleration[1] for acceleration in accelerations]), np.array(
        [acceleration[2] for acceleration in accelerations])


def analysis(df_angles: DataFrame, pts_cam: list, analysis_keypoint):
    sensormotionDrawing = []
    frames_time_sec_raw = df_angles["Time_in_sec"].to_list()
    """
    3D空间下对指定analysis_keypoint的加速度进行分解
    """
    accelerations_x, accelerations_y, accelerations_z = calculateAccelerationListFrame(
        [keypoints[analysis_keypoint.value] for keypoints in pts_cam], frames_time_sec_raw)
    sampling_rate = 10
    frames_time_sec = np.array(frames_time_sec_raw[1:])

    """
    绘制加速度视图
    """

    b, a = sm.signal.build_filter(3, sampling_rate, 'low', filter_order=4)

    # 滤波器过滤信号
    x_f = sm.signal.filter_signal(b, a, accelerations_x)  # ML medio-lateral
    y_f = sm.signal.filter_signal(b, a, accelerations_y)  # VT vertical
    z_f = sm.signal.filter_signal(b, a, accelerations_z)  # AP antero-posterior

    sensormotionDrawing.append(
        polt_accelerations(frames_time_sec, accelerations_x, accelerations_y, accelerations_z, x_f, y_f, z_f))

    """
    计算基于自相关的指标，例如步长规律性、步幅规律性和步长对称性。
    """
    ac, ac_lags, xcorr_drawing = plot_xcorr(y_f, y_f, scale='unbiased')
    sensormotionDrawing.append(xcorr_drawing)

    """
    提取步态指标，如步速、步长等, 识别信号中的谷
    """
    peak_times, peak_values, peak_drawing = polt_find_peaks(frames_time_sec, y_f, peak_type='valley', min_val=0.6,
                                                            min_dist=10)
    sensormotionDrawing.append(peak_drawing)
    step_count = sm.gait.step_count(peak_times)
    cadence = sm.gait.cadence(frames_time_sec, peak_times)
    step_time, step_time_sd, step_time_cov = sm.gait.step_time(peak_times)

    """
    计算垂直 (Y) 信号的自相关 (AC)，检测 AC 中的峰值，然后计算步态指标
    """
    ac_peak_times, ac_peak_values, ac_peak_drawing = polt_find_peaks(ac_lags, ac, peak_type='peak', min_val=0.1,
                                                                     min_dist=30)
    sensormotionDrawing.append(ac_peak_drawing)
    step_reg, stride_reg = sm.gait.step_regularity(ac_peak_values)
    step_sym = sm.gait.step_symmetry(ac_peak_values)

    """
    根据 MVPA 文献中的预定义切点将身体活动的时期（窗口）分类为不同的强度级别
    使用预定义的切点集将每个时期分类为 PA 强度（例如久坐、中等、剧烈）。 
    切点集是已发表研究文章的计数阈值，此包中包含多个集（用于不同人群）。
    """

    x_counts = sm.pa.convert_counts(accelerations_x, frames_time_sec, time_scale='s', epoch=1, rectify='full',
                                    integrate='simpson', plot=False)
    y_counts = sm.pa.convert_counts(accelerations_y, frames_time_sec, time_scale='s', epoch=1, rectify='full',
                                    integrate='simpson', plot=False)
    z_counts = sm.pa.convert_counts(accelerations_z, frames_time_sec, time_scale='s', epoch=1, rectify='full',
                                    integrate='simpson', plot=False)

    vm = sm.signal.vector_magnitude(x_counts, y_counts, z_counts)

    categories, time_spent, cut_points_drawing = plot_cut_points(vm, set_name='butte_preschoolers', n_axis=3)
    sensormotionDrawing.append(cut_points_drawing)

    return {"SpatiotemporalData": [
        ["参数Parameters", "数值Data", "单位Unit", "参考值Reference"],
        ["Number of step\n步数", str(step_count), "-", "-"],
        ["Cadence\n步频", str(round(cadence / 60 / 1000, 2)), "steps/sec", "2.274±0.643"],
        ["Stride time\n跨步时间", str((step_time / 1000).round(2)), "sec", "0.901±0.293"],
        ["Step time variability(SD)\n步长时间变化(标准差)", str(step_time_sd.round(2)), "-", "-"],
        ["Step time variability(CoV)\n步长时间变化系数", str((step_time_cov * 100).round(2)), "CoV(%)", "22.847±22.72"],
        ["Step regularity\n步长规律指数", str(step_reg.round(4)), "-", "-"],
        ["Stride regularity\n步幅规律指数", str(stride_reg.round(4)), "-", "-"],
        ["Step symmetry\n步长对称指数", str(step_sym.round(4)), "-", "-"],

        ["Total\n样本步行总时长", str(df_angles["Time_in_sec"].iloc[-1].round(2)), "sec", "-"],

        ["检测项共计", "", "", "9 项"]
    ], "SpatiotemporalGraph": sensormotionDrawing}
