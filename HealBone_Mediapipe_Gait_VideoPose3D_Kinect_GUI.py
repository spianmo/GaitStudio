import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, NoReturn, List, Tuple, Any

import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from matplotlib import pyplot as plt
from mediapipe.python.solutions.pose import PoseLandmark

from numpy import ndarray
import seaborn as sns
from pandas import DataFrame
from pyk4a import PyK4A, Config, ColorResolution, FPS, DepthMode
import pyk4a

import Gait_Analysis
import MainWindow
from GUISignal import LogSignal
from acceleration import sensormotionDemo
from estimator.estimator_test import simple_plot_angles
from estimator.utils.angle_helper import calc_common_angles
from estimator.videopose3d_async import VideoPose3DAsync
from kinect_helpers import depthInMeters, color_depth_image, colorize, smooth_depth_image, obj2json
from kinect_smoothing import Denoising_Filter, HoleFilling_Filter
from MainWindow import Ui_MainWindow
from widgets.CTitleBar import CTitleBar

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

frame_shape = [1080, 1920]

# 待检测的点
checked_pose_keypoints = [
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_EYE_INNER,
    PoseLandmark.LEFT_EYE,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_INNER,
    PoseLandmark.RIGHT_EYE,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT,
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_PINKY,
    PoseLandmark.RIGHT_PINKY,
    PoseLandmark.LEFT_INDEX,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.LEFT_THUMB,
    PoseLandmark.RIGHT_THUMB,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.LEFT_HEEL,
    PoseLandmark.RIGHT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX,
    PoseLandmark.RIGHT_FOOT_INDEX,
]


def BGR(RGB: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    RGB Color to BGR Color
    :param RGB: RGB Color
    :return: BGR Color
    """
    return RGB[2], RGB[1], RGB[0]


def draw_healbone_logo(frame: ndarray) -> NoReturn:
    """
    add HealBone Logo to CV-Frame
    """
    logo = cv.imread('./logo.png')
    width = 123 * 2
    height = int(width / 4.3)
    logo = cv.resize(logo, (width, height))
    img2gray = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 1, 255, cv.THRESH_BINARY)
    roi = frame[-height - 10:-10, -width - 10:-10]
    roi[np.where(mask)] = 0
    roi += logo


def plot_angles(title: str, df: DataFrame) -> None:
    rc = {'font.sans-serif': 'SimHei',
          'axes.unicode_minus': False}

    sns.set_style(style='darkgrid', rc=rc)

    fig, axes = plt.subplots(3, 4, figsize=(24, 14))

    # axes[0, 0].set(ylim=(20, 160))
    # axes[0, 1].set(ylim=(20, 160))
    # axes[0, 2].set(ylim=(0, 100))
    # axes[0, 3].set(ylim=(0, 100))
    # axes[1, 0].set(ylim=(20, 160))
    # axes[1, 1].set(ylim=(20, 160))
    # axes[1, 2].set(ylim=(20, 180))
    # axes[1, 3].set(ylim=(20, 180))
    # axes[2, 0].set(ylim=(0, 90))
    # axes[2, 1].set(ylim=(0, 90))

    axes[0, 0].set(ylim=(0, 180))
    axes[0, 1].set(ylim=(0, 180))
    axes[0, 2].set(ylim=(0, 180))
    axes[0, 3].set(ylim=(0, 180))
    axes[1, 0].set(ylim=(0, 180))
    axes[1, 1].set(ylim=(0, 180))
    axes[1, 2].set(ylim=(0, 180))
    axes[1, 3].set(ylim=(0, 180))
    axes[2, 0].set(ylim=(0, 180))
    axes[2, 1].set(ylim=(0, 180))

    fig.suptitle("关节角度变化周期 - " + title)

    sns.lineplot(ax=axes[0, 0], data=df, x="Time_in_sec", y="TorsoLHip_angle").set(xlabel="时间（秒）",
                                                                                   ylabel="躯干 L 髋关节角度 (°)")
    sns.lineplot(ax=axes[0, 1], data=df, x="Time_in_sec", y="TorsoRHip_angle").set(xlabel="时间（秒）",
                                                                                   ylabel="躯干 R 髋关节角度 (°)")

    sns.lineplot(ax=axes[0, 2], data=df, x="Time_in_sec", y="TorsoLFemur_angle").set(xlabel="时间（秒）",
                                                                                     ylabel="L 髋关节角度（屈曲伸展） (°)")
    sns.lineplot(ax=axes[0, 3], data=df, x="Time_in_sec", y="TorsoRFemur_angle").set(xlabel="时间（秒）",
                                                                                     ylabel="R 髋关节角度（屈曲伸展）(°)")

    sns.lineplot(ax=axes[1, 0], data=df, x="Time_in_sec", y="LHip_angle").set(xlabel="时间（秒）",
                                                                              ylabel="L 髋关节角度（内收外展） (°)")
    sns.lineplot(ax=axes[1, 1], data=df, x="Time_in_sec", y="RHip_angle").set(xlabel="时间（秒）",
                                                                              ylabel="R 髋关节角度（内收外展） (°)")

    sns.lineplot(ax=axes[1, 2], data=df, x="Time_in_sec", y="LTibiaSelf_vector").set(xlabel="时间（秒）",
                                                                                     ylabel="L 髋关节角度（外旋内旋） (°)")
    sns.lineplot(ax=axes[1, 3], data=df, x="Time_in_sec", y="RTibiaSelf_vector").set(xlabel="时间（秒）",
                                                                                     ylabel="R 髋关节角度（外旋内旋） (°)")

    sns.lineplot(ax=axes[2, 0], data=df, x="Time_in_sec", y="LKnee_angle").set(xlabel="时间（秒）",
                                                                               ylabel="L 膝关节角度 (°)")
    sns.lineplot(ax=axes[2, 1], data=df, x="Time_in_sec", y="RKnee_angle").set(xlabel="时间（秒）",
                                                                               ylabel="R 膝关节角度 (°)")

    plt.show()


def vectors_to_angle(vector1, vector2) -> float:
    """
    计算两个向量之间的夹角
    :param vector1:
    :param vector2:
    :return:
    """
    x = np.dot(vector1, -vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    theta = np.degrees(np.arccos(x))
    return 180 - theta


def build_vector_h36m(landmarks, landmark_index) -> ndarray:
    """
    根据关节点坐标构建向量
    :param landmark_index:
    :param landmarks:
    :return:
    """
    return np.array(
        [landmarks[landmark_index][0], landmarks[landmark_index][1], landmarks[landmark_index][2]])


def build_vector_mediapipe(landmarks, landmark_index: PoseLandmark) -> ndarray:
    """
    根据关节点坐标构建向量
    :param landmark_index:
    :param landmarks:
    :return:
    """
    return np.array(
        [landmarks[landmark_index.value].x, landmarks[landmark_index.value].y, landmarks[landmark_index.value].z])


def landmark_to_angle_h36m(landmarks) -> dict:
    """
    计算单次姿态的所有点的检测夹角
    :param landmarks:
    :return:
    """
    MHip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, Spine1, Neck, \
    Head, Site, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist = range(17)
    # 鼻部坐标
    Nose_coor = build_vector_h36m(landmarks, Head)
    # 左髋关节坐标
    LHip_coor = build_vector_h36m(landmarks, LHip)
    # 右髋关节坐标
    RHip_coor = build_vector_h36m(landmarks, RHip)
    # 左右髋关节中点
    MidHip_coor = build_vector_h36m(landmarks, MHip)
    # 左膝关节坐标
    LKnee_coor = build_vector_h36m(landmarks, LKnee)
    # 右膝关节坐标
    RKnee_coor = build_vector_h36m(landmarks, RKnee)
    # 左踝关节坐标
    LAnkle_coor = build_vector_h36m(landmarks, LAnkle)
    # 右踝关节坐标
    RAnkle_coor = build_vector_h36m(landmarks, RAnkle)

    # 躯干向量
    Torso_vector = MidHip_coor - Nose_coor
    # 左右胯骨向量
    Hip_vector = LHip_coor - RHip_coor
    # 左股骨向量
    LFemur_vector = LKnee_coor - LHip_coor
    # 右股骨向量
    RFemur_vector = RKnee_coor - RHip_coor
    # 左胫骨向量
    LTibia_vector = LAnkle_coor - LKnee_coor
    # 右胫骨向量
    RTibia_vector = RAnkle_coor - RKnee_coor

    # 躯干与胯骨的夹角
    TorsoLHip_angle = vectors_to_angle(Torso_vector, Hip_vector)
    TorsoRHip_angle = vectors_to_angle(Torso_vector, -Hip_vector)

    # 内收外展
    # 左股骨与胯骨的夹角
    LHip_angle = vectors_to_angle(LFemur_vector, Hip_vector)
    # 右股骨与胯骨的夹角
    RHip_angle = vectors_to_angle(RFemur_vector, -Hip_vector)

    # 屈曲伸展
    # 躯干与股骨
    TorsoLFemur_angle = vectors_to_angle(Torso_vector, LFemur_vector)
    TorsoRFemur_angle = vectors_to_angle(Torso_vector, RFemur_vector)

    # 外旋内旋
    # 胫骨旋转
    LTibiaSelf_vector = vectors_to_angle(LTibia_vector, np.array([0, 1, 0]))
    RTibiaSelf_vector = vectors_to_angle(RTibia_vector, np.array([0, 1, 0]))

    # 左胫骨与左股骨的夹角
    LKnee_angle = vectors_to_angle(LTibia_vector, LFemur_vector)
    # 右胫骨与右股骨的夹角
    RKnee_angle = vectors_to_angle(RTibia_vector, RFemur_vector)

    dict_angles = {"TorsoLHip_angle": TorsoLHip_angle, "TorsoRHip_angle": TorsoRHip_angle, "LHip_angle": LHip_angle,
                   "RHip_angle": RHip_angle, "LKnee_angle": LKnee_angle, "RKnee_angle": RKnee_angle,
                   "TorsoLFemur_angle": TorsoLFemur_angle, "TorsoRFemur_angle": TorsoRFemur_angle,
                   "LTibiaSelf_vector": LTibiaSelf_vector, "RTibiaSelf_vector": RTibiaSelf_vector}
    return dict_angles


def landmark_to_angle_mediapipe(landmarks) -> dict:
    """
    计算单次姿态的所有点的检测夹角
    :param landmarks:
    :return:
    """
    # 鼻部坐标
    Nose_coor = build_vector_h36m(landmarks, mp_pose.PoseLandmark.NOSE)
    # 左髋关节坐标
    LHip_coor = build_vector_h36m(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    # 右髋关节坐标
    RHip_coor = build_vector_h36m(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
    # 左右髋关节中点
    MidHip_coor = np.array(
        [(LHip_coor[0] + RHip_coor[0]) / 2, (LHip_coor[1] + RHip_coor[1]) / 2, (LHip_coor[2] + RHip_coor[2]) / 2])
    # 左膝关节坐标
    LKnee_coor = build_vector_h36m(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
    # 右膝关节坐标
    RKnee_coor = build_vector_h36m(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
    # 左踝关节坐标
    LAnkle_coor = build_vector_h36m(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
    # 右踝关节坐标
    RAnkle_coor = build_vector_h36m(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)

    # 躯干向量
    Torso_vector = MidHip_coor - Nose_coor
    # 左右胯骨向量
    Hip_vector = LHip_coor - RHip_coor
    # 左股骨向量
    LFemur_vector = LKnee_coor - LHip_coor
    # 右股骨向量
    RFemur_vector = RKnee_coor - RHip_coor
    # 左胫骨向量
    LTibia_vector = LAnkle_coor - LKnee_coor
    # 右胫骨向量
    RTibia_vector = RAnkle_coor - RKnee_coor

    # 躯干与胯骨的夹角
    TorsoLHip_angle = vectors_to_angle(Torso_vector, Hip_vector)
    TorsoRHip_angle = vectors_to_angle(Torso_vector, -Hip_vector)

    # 内收外展
    # 左股骨与胯骨的夹角
    LHip_angle = vectors_to_angle(LFemur_vector, Hip_vector)
    # 右股骨与胯骨的夹角
    RHip_angle = vectors_to_angle(RFemur_vector, -Hip_vector)

    # 屈曲伸展
    # 躯干与股骨
    TorsoLFemur_angle = vectors_to_angle(Torso_vector, LFemur_vector)
    TorsoRFemur_angle = vectors_to_angle(Torso_vector, RFemur_vector)

    # 外旋内旋
    # 胫骨旋转
    LTibiaSelf_vector = vectors_to_angle(LTibia_vector, np.array([0, 1, 0]))
    RTibiaSelf_vector = vectors_to_angle(RTibia_vector, np.array([0, 1, 0]))

    # 左胫骨与左股骨的夹角
    LKnee_angle = vectors_to_angle(LTibia_vector, LFemur_vector)
    # 右胫骨与右股骨的夹角
    RKnee_angle = vectors_to_angle(RTibia_vector, RFemur_vector)

    dict_angles = {"TorsoLHip_angle": TorsoLHip_angle, "TorsoRHip_angle": TorsoRHip_angle, "LHip_angle": LHip_angle,
                   "RHip_angle": RHip_angle, "LKnee_angle": LKnee_angle, "RKnee_angle": RKnee_angle,
                   "TorsoLFemur_angle": TorsoLFemur_angle, "TorsoRFemur_angle": TorsoRFemur_angle,
                   "LTibiaSelf_vector": LTibiaSelf_vector, "RTibiaSelf_vector": RTibiaSelf_vector}
    return dict_angles


def credible_pose(pose_keypoints):
    confidences = [keypoint[3] for keypoint in pose_keypoints]
    return np.array(confidences).min() > 0.5


global recording, record_frame_count


def show_cv_frame(cameraView, frame):
    """
    将cv的frame显示到label上
    :param cameraView:
    :param frame:
    """
    shrink = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    QtImg = QImage(shrink.data,
                   shrink.shape[1],
                   shrink.shape[0],
                   shrink.shape[1] * 3,
                   QImage.Format_RGB888)
    jpg_out = QPixmap(QtImg).scaled(cameraView.width(), cameraView.height(), Qt.KeepAspectRatio)
    scene = QGraphicsScene()  # 创建场景
    scene.addItem(QGraphicsPixmapItem(jpg_out))
    cameraView.setScene(scene)  # 将场景添加至视图


def read_video_frames(k4a, time, fps, videoFrameHandler: Callable[[tuple], tuple], poseLandmarksProtoCallback: Callable,
                      poseLandmarksCallback: Callable) -> tuple:
    """
    从视频流中读取帧，并将帧传递给回调函数
    :param k4a:
    :param poseLandmarksCallback:
    :param poseLandmarksProtoCallback:
    :param videoFrameHandler:
    :returns
    """

    pts_cams: List = []
    pts_3d: list = []

    while True:
        capture = k4a.get_capture()
        # 原始的RGBA视频帧
        frame = capture.color[:, :, :3]

        depth_image_raw = capture.transformed_depth

        # OpenCV自带的去噪修复，帧率太低
        # depth_image_raw = smooth_depth_image(depth_image_raw, max_hole_size=10)

        # 孔洞填充滤波器
        # hole_filter = HoleFilling_Filter(flag='min')
        # depth_image_raw = hole_filter.smooth_image(depth_image_raw)
        # 去噪滤波器
        # noise_filter = Denoising_Filter(flag='modeling', theta=60)
        # depth_image_raw = noise_filter.smooth_image(depth_image_raw)

        # 深度图像数据归一化为米
        depth_image = depthInMeters(depth_image_raw)
        if np.any(capture.depth):

            # 将BGR转换为RGB
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # 提升性能，不写入内存
            frame.flags.writeable = False
            pose_landmarks, pose_world_landmarks, pose_landmarks_proto, pose_world_landmarks_proto = videoFrameHandler(frame)
            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            if pose_landmarks is None or pose_world_landmarks is None or pose_landmarks_proto is None or \
                    pose_world_landmarks_proto is None:
                continue
            # 将归一化的坐标转换为原始坐标
            pose_keypoints = []

            for pose_landmark_index, pose_landmark in enumerate(pose_landmarks):
                if pose_landmark:
                    # 只处理待检测的关键点，用于后续CheckCube扩展
                    if PoseLandmark(pose_landmark_index) not in checked_pose_keypoints:
                        continue
                    visualize_x = int(round(pose_landmark.x * frame.shape[1]))
                    visualize_y = int(round(pose_landmark.y * frame.shape[0]))
                    truth_x = pose_landmark.x
                    truth_y = pose_landmark.y
                    # MediaPipe原始的landmark_z不可信
                    truth_z = pose_landmark.z
                    deep_axis1 = visualize_y if visualize_y < depth_image.shape[0] else depth_image.shape[0] - 1
                    deep_axis2 = visualize_x if visualize_x < depth_image.shape[1] else depth_image.shape[1] - 1
                    deep_z = depth_image[deep_axis1 if deep_axis1 > 0 else 0,
                                         deep_axis2 if deep_axis2 > 0 else 0]
                    visibility = pose_landmark.visibility
                    cv.putText(frame, "Depth:" + str(
                        round(deep_z, 3)),
                               (visualize_x - 10, visualize_y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, BGR(RGB=(102, 153, 250)), 1,
                               cv.LINE_AA)
                    cv.circle(frame, (visualize_x, visualize_y), radius=3, color=BGR(RGB=(255, 0, 0)), thickness=-1)
                    pose_keypoints.append([truth_x, truth_y, deep_z, visibility])
                else:
                    pose_keypoints = [[-1, -1, -1, -1]] * len(checked_pose_keypoints)

            global recording, record_frame_count
            global detectStatus
            if not detectStatus or record_frame_count == (fps * time):
                k4a.stop()
                break
            if credible_pose(pose_keypoints):
                if not recording:
                    logSignal.signal.emit("已识别到所有检测点，开始检测")
                    print("开始检测！")
                    recording = True
                pts_cams.append(pose_keypoints)
                record_frame_count += 1
                logSignal.signal.emit("已检测" + str((record_frame_count * (1 / fps))) + '秒')
                print("已检测" + str(record_frame_count * (1 / fps)) + '秒')
            else:
                if recording:
                    recording = False
                    record_frame_count = 0
                    pts_cams = []
                    logSignal.signal.emit("检测过程被打断！等待重新检测")
                    print("检测过程被打断！等待重新检测")
                    continue

            if poseLandmarksCallback:
                poseLandmarksCallback(pose_keypoints)

            if poseLandmarksProtoCallback:
                poseLandmarksProtoCallback(pose_landmarks_proto, frame, color_depth_image(depth_image_raw))

            k = cv.waitKey(1)
            # 按ESC键退出
            if k & 0xFF == 27:
                k4a.stop()
                break

        del capture
    cv.destroyAllWindows()
    hbWin.stopDetect()
    return pts_cams, np.array(pts_3d), fps


global poseDetector


def infer_pose(video_frame) -> Any:
    """
    推断视频帧中人体姿态
    :param video_frame:
    :return
    """
    global poseDetector
    return poseDetector.process(video_frame)


def video_frame_handler(video_frame):
    """
    每一帧视频帧被读取到时的异步Handler
    :param video_frame:
    :returns: pose_landmarks, pose_world_landmarks, pose_landmarks_proto, pose_world_landmarks_proto
    """
    infer_result = infer_pose(video_frame)

    if infer_result.pose_landmarks is None or infer_result.pose_world_landmarks is None:
        return None, None, None, None
    pose_landmarks = infer_result.pose_landmarks.landmark
    pose_world_landmarks = infer_result.pose_world_landmarks.landmark

    pose_landmarks_proto = infer_result.pose_landmarks
    pose_world_landmarks_proto = infer_result.pose_world_landmarks

    return pose_landmarks, pose_world_landmarks, pose_landmarks_proto, pose_world_landmarks_proto


def pose_landmarks_proto_handler(pose_landmarks_proto, frame, deep_frame, cameraView):
    """
    多source姿态关键点proto回调函数
    :param deep_frame:
    :param pose_landmarks_proto:
    :param frame:
    """
    combined_image = cv.addWeighted(frame, 0.5, colorize(deep_frame), 0.5, 0)
    mp_drawing.draw_landmarks(combined_image, pose_landmarks_proto, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    mp_drawing.draw_landmarks(frame, pose_landmarks_proto, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    mp_drawing.draw_landmarks(deep_frame, pose_landmarks_proto, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # 绘制HealBone图标
    draw_healbone_logo(combined_image)
    draw_healbone_logo(frame)

    # 窗口展示视频帧
    show_cv_frame(cameraView[0], cv.resize(frame, (0, 0), fx=0.6, fy=0.6))
    show_cv_frame(cameraView[1], cv.resize(colorize(deep_frame, (None, 5000), cv.COLORMAP_HSV), (0, 0), fx=0.6, fy=0.6))
    show_cv_frame(cameraView[2], cv.resize(combined_image, (0, 0), fx=0.6, fy=0.6))
    # cv.imshow("HealBone-Mediapipe-Gait: KinectCamera FOV", cv.resize(frame, (0, 0), fx=0.6, fy=0.6))
    # cv.imshow("HealBone-Mediapipe-Gait: KinectCamera IR", cv.resize(colorize(deep_frame, (None, 5000), cv.COLORMAP_HSV), (0, 0), fx=0.6, fy=0.6))
    # cv.imshow("HealBone-Mediapipe-Gait: KinectCamera FOV IR", cv.resize(combined_image, (0, 0), fx=0.6, fy=0.6))


def pose_landmarks_handler(pose_landmarks):
    """
    多source姿态关键点回调函数
    :param pose_landmarks_index:
    :param pose_landmarks:
    """
    # print(pose_landmarks)
    # logSignal.signal.emit(str(pose_landmarks))


def save_pts(filename: str, pts: ndarray) -> NoReturn:
    pts_output = Path("pts_output")
    if not pts_output.is_dir():
        os.makedirs(pts_output)
    file = open('pts_output/' + filename, "w")
    json.dump(pts.tolist(), file)
    file.close()


def main(show_plot_angle_demo, k4a, time, fps, cameraView):
    use_video_pose_3d = False

    global recording, record_frame_count

    recording = False
    record_frame_count = 0
    # opencv读取视频source，并使用mediapipe进行KeyPoints推理
    pts_cams_ndarray, pts_3d_ndarray, fps = read_video_frames(k4a, time, fps, videoFrameHandler=lambda frame: video_frame_handler(frame),
                                                              poseLandmarksProtoCallback=lambda pose_landmarks_proto, frame, deep_frame:
                                                              pose_landmarks_proto_handler(pose_landmarks_proto, frame, deep_frame, cameraView),
                                                              poseLandmarksCallback=lambda pose_landmarks:
                                                              pose_landmarks_handler(pose_landmarks))
    logSignal.signal.emit("开始2D->3D人体姿势姿势估计计算")
    if use_video_pose_3d:
        # 使用videoPose估计器
        logSignal.signal.emit("使用FaceBook VideoPose3D姿势估计器")
        estimator_3d = VideoPose3DAsync()
        videopose3d_pose = estimator_3d.estimate(pts_cams_ndarray, fps, w=frame_shape[1], h=frame_shape[0])
    else:
        logSignal.signal.emit("未使用FaceBook VideoPose姿势估计器，使用Kinect Depth深度数据")
        videopose3d_pose = pts_cams_ndarray

    # xx = np.array(pts_cams_ndarray)

    chart_data: list = []

    logSignal.signal.emit("正在计算每一帧的3D坐标中的角度")
    for pose_landmark_index, pose_landmark in enumerate(videopose3d_pose):
        # 计算每一帧的3D坐标中的角度
        angle_dict = landmark_to_angle_h36m(pose_landmark) if use_video_pose_3d else landmark_to_angle_mediapipe(pose_landmark)
        chart_data.append(angle_dict)

    logSignal.signal.emit("角度序列计算完成")
    df_angles = pd.DataFrame(chart_data)

    if df_angles.size == 0:
        logSignal.signal.emit("未检测到步态周期角度序列, 请重新开始检测")
        return
    df_angles = pd.DataFrame({"TorsoLHip_angle": df_angles["TorsoLHip_angle"], "TorsoRHip_angle": df_angles["TorsoRHip_angle"],
                              "LHip_angle": df_angles["LHip_angle"],
                              "RHip_angle": df_angles["RHip_angle"], "LKnee_angle": df_angles["LKnee_angle"],
                              "RKnee_angle": df_angles["RKnee_angle"],
                              "TorsoLFemur_angle": df_angles["TorsoLFemur_angle"],
                              "TorsoRFemur_angle": df_angles["TorsoRFemur_angle"],
                              "LTibiaSelf_vector": df_angles["LTibiaSelf_vector"],
                              "RTibiaSelf_vector": df_angles["RTibiaSelf_vector"]})
    df_angles["Time_in_sec"] = [n / fps for n in range(len(df_angles))]
    if show_plot_angle_demo:
        plot_angles("CAM[Fixed]", pd.DataFrame(df_angles))
    logSignal.signal.emit("检测序列已分析为角度序列")
    logSignal.signal.emit(df_angles.to_markdown())

    # 分析步态周期
    logSignal.signal.emit("正在分析步态周期")
    Gait_Analysis.analysis(df_angles=pd.DataFrame(df_angles), fps=fps, pts_cam=pts_cams_ndarray, analysis_keypoint=PoseLandmark.RIGHT_KNEE)
    logSignal.signal.emit("分析报告已生成")
    # plt.show()


def start(time, cameraView, k4aConfig, mpConfig):
    show_plot_angle_demo = True
    k4a = PyK4A(
        k4aConfig
    )
    logSignal.signal.emit(str(obj2json(k4aConfig)))
    k4a.start()
    global poseDetector
    try:
        if poseDetector:
            poseDetector.reset()
    except:
        poseDetector = None

    if not poseDetector:
        poseDetector = mp_pose.Pose(
            min_detection_confidence=mpConfig["min_detection_confidence"],
            min_tracking_confidence=mpConfig["min_tracking_confidence"],
            model_complexity=mpConfig["model_complexity"],
            smooth_landmarks=mpConfig["smooth_landmarks"]
        )
    main(show_plot_angle_demo=show_plot_angle_demo, k4a=k4a, time=time, fps=30, cameraView=cameraView)


global detectStatus


class HealBoneWindow(QMainWindow, MainWindow.Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        MainWindow.Ui_MainWindow.__init__(self)
        self.isInit = False
        # self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setupUi(self)
        self.tabifyDockWidget(self.viewerDock, self.angleViewerDock)
        self.btnStart.clicked.connect(self.btnStartClicked)

    def resizeCameraView(self, tabWidth=-1, tabHeight=-1):
        tabWidgetSize: QSize = self.tabWidget.geometry().size()
        self.cameraIrFovView.setGeometry(self.cameraIrFovView.x(), self.cameraIrFovView.y(), tabWidgetSize.width() if tabWidth == -1 else tabWidth,
                                         (tabWidgetSize.height() - 24) if tabHeight == -1 else tabHeight)
        self.cameraFovView.setGeometry(self.cameraFovView.x(), self.cameraFovView.y(), tabWidgetSize.width() if tabWidth == -1 else tabWidth,
                                       (tabWidgetSize.height() - 24) if tabHeight == -1 else tabHeight)
        self.cameraIrView.setGeometry(self.cameraIrView.x(), self.cameraIrView.y(), tabWidgetSize.width() if tabWidth == -1 else tabWidth,
                                      (tabWidgetSize.height() - 24) if tabHeight == -1 else tabHeight)

    def resizeEvent(self, event: QResizeEvent):
        if self.isInit:
            self.resizeCameraView()
        else:
            self.isInit = True
            self.resizeCameraView(740, 535)

    def stopDetect(self):
        global detectStatus
        detectStatus = False
        self.btnStart.setText("开始检测")

    def startDetect(self, k4aConfig, mpConfig):
        global detectStatus
        detectStatus = True
        self.btnStart.setText("停止检测")
        start(int(self.sbTime.text()), [self.cameraFovView, self.cameraIrView, self.cameraIrFovView], k4aConfig, mpConfig)

    def btnStartClicked(self):
        global detectStatus
        if detectStatus:
            self.stopDetect()
        else:
            k4aConfig = Config(
                color_resolution=ColorResolution(self.cbColorResolution.currentIndex() + 1),
                camera_fps=FPS(self.cbFPS.currentIndex()),
                depth_mode=DepthMode(self.cbDepthMode.currentIndex() + 1),
                synchronized_images_only=True,
            )
            mpConfig = {
                "min_detection_confidence": round(self.hsMinDetectionConfidence.sliderPosition() / self.hsMinDetectionConfidence.maximum(), 1),
                "min_tracking_confidence": round(self.hsMinTrackingConfidence.sliderPosition() / self.hsMinTrackingConfidence.maximum(), 1),
                "model_complexity": self.cbModelComplexity.currentIndex(),
                "smooth_landmarks": self.cbSmoothLandmarks.isChecked()
            }
            self.startDetect(k4aConfig, mpConfig)

    def logViewAppend(self, text):
        self.outputText.moveCursor(QTextCursor.End, QTextCursor.MoveMode.MoveAnchor)
        local_time_asctimes = time.strftime("%Y-%m-%d %H:%M:%S ==> ", time.localtime(time.time()))
        self.outputText.setMarkdown(
            self.outputText.toMarkdown(QTextDocument.MarkdownFeature.MarkdownDialectGitHub) + local_time_asctimes + text + '\n')
        if len(self.outputText.toHtml()) > 1024 * 1024 * 10:
            self.outputText.clear()
        scrollbar: QScrollBar = self.outputText.verticalScrollBar()
        if scrollbar:
            scrollbar.setSliderPosition(scrollbar.maximum())


if __name__ == '__main__':
    detectStatus = False
    app = QApplication()
    app.setStyleSheet(open('resources/styleSheet.qss', encoding='utf-8').read())
    hbWin = HealBoneWindow()
    # 信号槽
    logSignal = LogSignal()
    logSignal.signal.connect(lambda log: hbWin.logViewAppend(log))
    logSignal.signal.emit("HealBone GaitLab 初始化完成")

    hbWin.show()
    sys.exit(app.exec_())
