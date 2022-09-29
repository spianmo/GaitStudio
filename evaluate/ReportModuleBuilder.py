import time
from io import BytesIO
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from svglib.svglib import svg2rlg

from evaluate.EvaluateCore import PartAngle
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

plt.ioff()


def get_local_format_time(timestamp):
    local_time = time.localtime()
    format_time = time.strftime("%Y%m%d%H%M%S", local_time)
    return format_time


def generateROMPart(df_angles: pd.DataFrame, parts: list):
    romPart = []
    for part in parts:
        if part == PartAngle.Knee:
            romPart.append({
                "title": "膝关节活动度",
                "list": [
                    ["参数Parameters", "数值Data", "单位Unit", "参考值Reference"],
                    ["左膝关节伸展\nL.KNEE Extension", str(df_angles["LKnee_angle"].min().round(2)), "°", "0-60"],
                    ["左膝关节屈曲\nL.KNEE Flexion", str(df_angles["LKnee_angle"].max().round(2)), "°", "0-140"],
                    ["右膝关节伸展\nR.KNEE Extension", str(df_angles["RKnee_angle"].min().round(2)), "°", "0-60"],
                    ["右膝关节屈曲\nR.KNEE Flexion", str(df_angles["RKnee_angle"].max().round(2)), "°", "0-140"],
                    ["检测项共计", "", "", "4 项"]
                ]
            })
        elif part == PartAngle.Hip:
            romPart.append({
                "title": "髋关节活动度",
                "list": [
                    ["参数Parameters", "数值Data", "单位Unit", "参考值Reference"],
                    ["左髋关节伸展\nL.Hip Extension", str(df_angles["TorsoLFemur_angle"].min().round(2)), "°", "0-30"],
                    ["左髋关节屈曲\nL.Hip Flexion", str(df_angles["TorsoLFemur_angle"].max().round(2)), "°", "0-40"],
                    ["右髋关节伸展\nR.Hip Extension", str(df_angles["TorsoRFemur_angle"].min().round(2)), "°", "0-30"],
                    ["右髋关节屈曲\nR.Hip Flexion", str(df_angles["TorsoRFemur_angle"].max().round(2)), "°", "0-40"],
                    ["左髋关节外展\nL.Hip Abduction", str((180 - df_angles["LHip_angle"].max() - 90).round(2)), "°",
                     "-"],
                    ["左髋关节内收\nL.Hip Adduction", str((90 - (180 - df_angles["LHip_angle"].min())).round(2)), "°",
                     "-"],
                    ["右髋关节外展\nR.Hip Abduction", str((180 - df_angles["RHip_angle"].max() - 90).round(2)), "°",
                     "-"],
                    ["右髋关节内收\nR.Hip Adduction", str((90 - (180 - df_angles["RHip_angle"].min())).round(2)), "°",
                     "-"],
                    ["左髋关节外旋\nL.Hip Internal Rotation",
                     str((180 - df_angles["LTibiaSelf_vector"].max()).round(2)),
                     "°", "-"],
                    ["左髋关节内旋\nL.Hip External Rotation", str((df_angles["LTibiaSelf_vector"].min()).round(2)), "°",
                     "-"],
                    ["右髋关节外旋\nR.Hip Internal Rotation",
                     str((180 - df_angles["RTibiaSelf_vector"].max()).round(2)),
                     "°", "-"],
                    ["右髋关节内旋\nR.Hip External Rotation", str((df_angles["RTibiaSelf_vector"].min()).round(2)), "°",
                     "-"],
                    ["检测项共计", "", "", "12 项"]
                ]
            })
        elif part == PartAngle.Pelvis:
            romPart.append({
                "title": "骨盆活动度",
                "list": [
                    ["参数Parameters", "数值Data", "单位Unit", "参考值Reference"],
                    ["骨盆侧倾\nPelvis Obliquity", str((90 - df_angles["TorsoLHip_angle"].max()).round(2)), "°",
                     "0-10"],
                    ["骨盆旋转\nPelvis Rotation", str((90 - df_angles["TorsoLHip_angle"].min()).round(2)), "°", "0-10"],
                    ["检测项共计", "", "", "2 项"]
                ]
            })
        elif part == PartAngle.Ankle:
            romPart.append({
                "title": "踝关节活动度",
                "list": [
                    ["参数Parameters", "数值Data", "单位Unit", "参考值Reference"],
                    ["左踝关节跖屈\nL.Ankle Plantar flexion", str(df_angles["LAnkle_angle"].max().round(2)), "°", "20"],
                    ["左踝关节背屈\nL.Ankle Dorsiflexion", str(df_angles["LAnkle_angle"].min().round(2)), "°", "30"],
                    ["右踝关节跖屈\nR.Ankle Plantar flexion", str(df_angles["RAnkle_angle"].max().round(2)), "°", "20"],
                    ["右踝关节背屈\nR.Ankle Dorsiflexion", str(df_angles["RAnkle_angle"].min().round(2)), "°", "30"],
                    ["左踝关节外翻\nL.Ankle Pronation", "-", "°", "15"],
                    ["左踝关节内翻\nL.Ankle Supination", "-", "°", "35"],
                    ["右踝关节外翻\nR.Ankle Pronation", "-", "°", "15"],
                    ["右踝关节内翻\nR.Ankle Supination", "-", "°", "35"],
                    ["检测项共计", "", "", "8 项"]
                ]
            })
    return romPart


def polt_angle_plots(df: DataFrame) -> List[BytesIO]:
    metadatas = [
        {
            "title": "膝关节角度变化周期",
            "ylim": (0, 180),
            "axis": [
                ["Time_in_sec", "LKnee_angle", "时间（秒）", "L 膝关节角度 (°)"],
                ["Time_in_sec", "RKnee_angle", "时间（秒）", "R 膝关节角度 (°)"]
            ]
        },
        {
            "title": "髋关节角度变化周期（内收外展）",
            "ylim": (0, 180),
            "axis": [
                ["Time_in_sec", "LHip_angle", "时间（秒）", "L 髋关节角度 (°)"],
                ["Time_in_sec", "RHip_angle", "时间（秒）", "R 髋关节角度 (°)"]
            ]
        },
        {
            "title": "髋关节角度变化周期（屈曲伸展）",
            "ylim": (0, 180),
            "axis": [
                ["Time_in_sec", "TorsoLFemur_angle", "时间（秒）", "L 髋关节角度 (°)"],
                ["Time_in_sec", "TorsoRFemur_angle", "时间（秒）", "R 髋关节角度 (°)"]
            ]
        },
        {
            "title": "髋关节角度变化周期（外旋内旋）",
            "ylim": (0, 180),
            "axis": [
                ["Time_in_sec", "LTibiaSelf_vector", "时间（秒）", "L 髋关节角度 (°)"],
                ["Time_in_sec", "RTibiaSelf_vector", "时间（秒）", "R 髋关节角度 (°)"]
            ]
        },
        {
            "title": "躯干髋关节角度变化周期",
            "ylim": (0, 180),
            "axis": [
                ["Time_in_sec", "TorsoLHip_angle", "时间（秒）", "躯干 L 髋关节角度 (°)"],
                ["Time_in_sec", "TorsoRHip_angle", "时间（秒）", "躯干 R 髋关节角度 (°)"]
            ]
        },
        {
            "title": "踝关节角度变化周期",
            "ylim": (0, 180),
            "axis": [
                ["Time_in_sec", "LAnkle_angle", "时间（秒）", "L 踝关节角度 (°)"],
                ["Time_in_sec", "RAnkle_angle", "时间（秒）", "R 踝关节角度 (°)"]
            ]
        }
    ]
    images = []
    rc = {'font.sans-serif': 'SimHei',
          'axes.unicode_minus': False}
    sns.set_style(style='darkgrid', rc=rc)
    for metadata in metadatas:
        fig, axes = plt.subplots(2, 1, figsize=(5.5, 7))

        fig.suptitle(metadata["title"])

        axes[0].set(ylim=metadata["ylim"])
        axes[1].set(ylim=metadata["ylim"])

        sns.lineplot(ax=axes[0], data=df, x=metadata["axis"][0][0], y=metadata["axis"][0][1]).set(
            xlabel=metadata["axis"][0][2],
            ylabel=metadata["axis"][0][3])

        sns.lineplot(ax=axes[1], data=df, x=metadata["axis"][1][0], y=metadata["axis"][1][1]).set(
            xlabel=metadata["axis"][1][2],
            ylabel=metadata["axis"][1][3])
        image = BytesIO()
        fig.tight_layout()
        fig.savefig(image, format='svg')
        image.seek(0)
        images.append(svg2rlg(image))
    return images
