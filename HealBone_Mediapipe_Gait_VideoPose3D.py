import json
import os
import sys
from pathlib import Path
from typing import Callable, NoReturn, List, Tuple

import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mediapipe.python.solutions.pose import PoseLandmark

from numpy import ndarray
import seaborn as sns
from pandas import DataFrame

import Gait_Analysis
from acceleration import sensormotionDemo
from estimator.estimator_test import simple_plot_angles
from estimator.utils.angle_helper import calc_common_angles
from estimator.videopose3d_async import VideoPose3DAsync

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

poseDetectorPool = []
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


def draw_healbone_logo(*frames: List[ndarray]) -> NoReturn:
    """
    add HealBone Logo to CV-Frame
    """
    for _, frame in enumerate(*frames):
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

    axes[0, 0].set(ylim=(20, 160))
    axes[0, 1].set(ylim=(20, 160))
    axes[0, 2].set(ylim=(0, 100))
    axes[0, 3].set(ylim=(0, 100))
    axes[1, 0].set(ylim=(20, 160))
    axes[1, 1].set(ylim=(20, 160))
    axes[1, 2].set(ylim=(20, 180))
    axes[1, 3].set(ylim=(20, 180))
    axes[2, 0].set(ylim=(0, 90))
    axes[2, 1].set(ylim=(0, 90))

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


def build_vector(landmarks, landmark_index) -> ndarray:
    """
    根据关节点坐标构建向量
    :param landmark_index:
    :param landmarks:
    :return:
    """
    return np.array(
        [landmarks[landmark_index][0], landmarks[landmark_index][1], landmarks[landmark_index][2]])


def landmark_to_angle(landmarks) -> dict:
    """
    计算单次姿态的所有点的检测夹角
    :param landmarks:
    :return:
    """
    MHip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, Spine1, Neck, \
    Head, Site, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist = range(17)
    # 鼻部坐标
    Nose_coor = build_vector(landmarks, Head)
    # 左髋关节坐标
    LHip_coor = build_vector(landmarks, LHip)
    # 右髋关节坐标
    RHip_coor = build_vector(landmarks, RHip)
    # 左右髋关节中点
    MidHip_coor = build_vector(landmarks, MHip)
    # 左膝关节坐标
    LKnee_coor = build_vector(landmarks, LKnee)
    # 右膝关节坐标
    RKnee_coor = build_vector(landmarks, RKnee)
    # 左踝关节坐标
    LAnkle_coor = build_vector(landmarks, LAnkle)
    # 右踝关节坐标
    RAnkle_coor = build_vector(landmarks, RAnkle)

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


def read_video_frames(*streams: any, videoFrameHandler: Callable[[tuple], tuple], poseLandmarksProtoCallback: Callable,
                      poseLandmarksCallback: Callable, crop_video: bool) -> tuple:
    """
    从视频流中读取帧，并将帧传递给回调函数
    :param poseLandmarksCallback:
    :param poseLandmarksProtoCallback:
    :param streams:
    :param videoFrameHandler:
    :returns
    """
    caps = [cv.VideoCapture(stream) for stream in streams]

    fps = [cv.VideoCapture(stream).get(cv.CAP_PROP_FPS) for stream in streams]

    if np.std(fps) != 0.0:
        raise Exception('sources different fps')

    pts_cams: List[list] = [[] for _ in range(len(caps))]
    pts_3d: list = []

    print("read_video_sources:", len(caps))
    print("fps:", fps[0])

    for cap_index, cap in enumerate(caps):
        # 视频流的分辨率设置为1920x1080
        if cap_index == 0:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_shape[1])
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_shape[0])
        else:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_shape[0])
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_shape[1])

    while True:
        frames: List[ndarray] = []
        # 遍历caps
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                break
            if crop_video:
                frame = frame[:, frame_shape[1] // 2 - frame_shape[0] // 2:frame_shape[1] // 2 + frame_shape[0] // 2]

            # 将BGR转换为RGB
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # 提升性能，不写入内存
            frame.flags.writeable = False
            frames.append(frame)
        if len(frames) is not len(caps):
            print("Error: not all caps read, just {}/{}".format(len(frames), len(caps)))
            break

        pose_landmarks, pose_world_landmarks, pose_landmarks_proto, pose_world_landmarks_proto = videoFrameHandler(
            *frames)

        if pose_landmarks is None or pose_world_landmarks is None or pose_landmarks_proto is None or \
                pose_world_landmarks_proto is None:
            continue

        # 解除写入性能限制，将RGB转换为BGR
        for frame_index, frame in enumerate(frames):
            frames[frame_index].flags.writeable = True
            frames[frame_index] = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # 将归一化的坐标转换为原始坐标
        for pose_landmark_index, pose_landmark in enumerate(pose_landmarks):
            pose_keypoints = []
            if pose_landmark:
                for keypoint_index, landmark in enumerate(pose_landmark):
                    # 只处理待检测的关键点，用于后续CheckCube扩展
                    if PoseLandmark(keypoint_index) not in checked_pose_keypoints:
                        continue
                    visualize_x = int(round(landmark.x * frames[pose_landmark_index].shape[1]))
                    visualize_y = int(round(landmark.y * frames[pose_landmark_index].shape[0]))
                    truth_x = landmark.x
                    truth_y = landmark.y
                    truth_z = landmark.z
                    visibility = landmark.visibility
                    cv.circle(frames[pose_landmark_index], (visualize_x, visualize_y), radius=3,
                              color=BGR(RGB=(255, 0, 0)), thickness=-1)
                    pose_keypoints.append([truth_x, truth_y, truth_z, visibility])

            else:
                pose_keypoints = [[-1, -1, -1, -1]] * len(checked_pose_keypoints)

            pts_cams[pose_landmark_index].append(pose_keypoints)

            if poseLandmarksCallback:
                poseLandmarksCallback(pose_landmark_index, pose_keypoints)

        if poseLandmarksProtoCallback:
            poseLandmarksProtoCallback(pose_landmarks_proto, frames)

        k = cv.waitKey(1)
        # 按ESC键退出
        if k & 0xFF == 27:
            break

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    return [np.array(_pts_cam) for _pts_cam in pts_cams], np.array(pts_3d), fps[0]


def infer_pose(*video_frames: tuple) -> list:
    """
    推断视频帧中人体姿态
    :param video_frames:
    :return
    """
    global poseDetectorPool
    if len(poseDetectorPool) == 0:
        poseDetectorPool = [mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True,
            smooth_segmentation=True,
        ) for _ in range(len(video_frames))]
        print("PoseDetectorPool Size:", len(poseDetectorPool))

    results = []
    for i in range(len(video_frames)):
        results.append(poseDetectorPool[i].process(video_frames[i]))
    return results


def video_frame_handler(*video_frame: tuple) -> Tuple[any, any, any, any]:
    """
    每一帧视频帧被读取到时的异步Handler
    :param video_frame:
    :returns: pose_landmarks, pose_world_landmarks, pose_landmarks_proto, pose_world_landmarks_proto
    """
    infer_results = infer_pose(*video_frame)

    for i in range(len(infer_results)):
        if infer_results[i].pose_landmarks is None or infer_results[i].pose_world_landmarks is None:
            return None, None, None, None
    pose_landmarks = [i.pose_landmarks.landmark for i in infer_results]
    pose_world_landmarks = [i.pose_world_landmarks.landmark for i in infer_results]

    pose_landmarks_proto = [i.pose_landmarks for i in infer_results]
    pose_world_landmarks_proto = [i.pose_world_landmarks for i in infer_results]

    return pose_landmarks, pose_world_landmarks, pose_landmarks_proto, pose_world_landmarks_proto


def pose_landmarks_proto_handler(pose_landmarks_proto, frames):
    """
    多source姿态关键点proto回调函数
    :param pose_landmarks_proto:
    :param frames:
    """
    for pose_landmark_proto_index, pose_landmark_proto in enumerate(pose_landmarks_proto):
        mp_drawing.draw_landmarks(frames[pose_landmark_proto_index], pose_landmark_proto, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # 绘制HealBone图标
    draw_healbone_logo(frames)

    # 窗口展示视频帧
    for frame_index, frame in enumerate(frames):
        cv.imshow("HealBone-Mediapipe-Gait: Camera" + str(frame_index), cv.resize(frame, (0, 0), fx=0.5, fy=0.5))


def pose_landmarks_handler(pose_landmarks_index, pose_landmarks):
    """
    多source姿态关键点回调函数
    :param pose_landmarks_index:
    :param pose_landmarks:
    """
    print(pose_landmarks_index, pose_landmarks)


def save_pts(filename: str, pts: ndarray) -> NoReturn:
    pts_output = Path("pts_output")
    if not pts_output.is_dir():
        os.makedirs(pts_output)
    file = open('pts_output/' + filename, "w")
    json.dump(pts.tolist(), file)
    file.close()


if __name__ == '__main__':
    show_plot_angle_demo = True
    store_raw_pts = True
    debug_mode = False
    crop_video = False

    # 预先读取的不同视角视频
    if debug_mode:
        # input_stream = (
        #     'data/multi/Walking.54138969.mp4',
        #     'data/multi/Walking.55011271.mp4', 'data/multi/Walking.58860488.mp4', 'data/multi/Walking.60457274.mp4')
        input_stream = ('data/multi-virtual/Walking.camera1.mp4',)
    else:
        # input_stream = ('data/multi-virtual/Walking.camera1.mp4', 'data/multi-virtual/Walking.camera2.mp4')
        input_stream = ('data/multi-hb/Walking.camera1.mp4', 'data/multi-hb/Walking.camera2.mp4')
        # input_stream = ('data/multi-hb-syj/Walking.camera1.mp4', 'data/multi-hb-syj/Walking.camera2.mp4')
        # input_stream = ('data/multi/Walking.54138969.mp4', 'data/multi/Walking.55011271.mp4')
        # input_stream = ('data/multi-25fps/Walking.54138969.mp4', 'data/multi-25fps/Walking.55011271.mp4')

    # 读取相机串口编号
    if len(sys.argv) == 3:
        input_stream = (int(sys.argv[1]), int(sys.argv[2]))

    # opencv读取视频source，并使用mediapipe进行KeyPoints推理
    pts_cams_ndarray, pts_3d_ndarray, fps = read_video_frames(*input_stream,
                                                              videoFrameHandler=lambda *frames:
                                                              video_frame_handler(*frames),
                                                              poseLandmarksProtoCallback=lambda pose_landmarks_proto, frames:
                                                              pose_landmarks_proto_handler(pose_landmarks_proto, frames),
                                                              poseLandmarksCallback=lambda pose_landmarks_index, pose_landmarks:
                                                              pose_landmarks_handler(pose_landmarks_index, pose_landmarks),
                                                              crop_video=crop_video)

    estimator_3d = VideoPose3DAsync()

    videopose3d_poses = [[] for _ in range(len(pts_cams_ndarray))]
    for index, pts_cam in enumerate(pts_cams_ndarray):
        _pts_cam, _end = np.split(pts_cam, [-2], axis=2)
        videopose3d_poses[index] = estimator_3d.estimate(_pts_cam, fps, w=frame_shape[1], h=frame_shape[0])

    chart_datas: list = [[] for _ in range(len(pts_cams_ndarray))]

    for chart_index, chart_data in enumerate(chart_datas):
        for pose_landmark_index, pose_landmark in enumerate(videopose3d_poses[chart_index]):
            # 计算每一帧的3D坐标中的角度
            angle_dict = landmark_to_angle(pose_landmark)
            chart_datas[chart_index].append(angle_dict)

    df_angless: list = [{} for _ in range(len(pts_cams_ndarray))]
    # 绘制步态周期图表
    for chart_index, chart_data in enumerate(chart_datas):
        df_angless[chart_index] = pd.DataFrame(chart_data)

    df_angles = pd.DataFrame({"TorsoLHip_angle": df_angless[0]["TorsoLHip_angle"], "TorsoRHip_angle": df_angless[0]["TorsoRHip_angle"],
                              "LHip_angle": df_angless[0]["LHip_angle"],
                              "RHip_angle": df_angless[0]["RHip_angle"], "LKnee_angle": df_angless[1]["LKnee_angle"],
                              "RKnee_angle": df_angless[1]["RKnee_angle"],
                              "TorsoLFemur_angle": df_angless[1]["TorsoLFemur_angle"],
                              "TorsoRFemur_angle": df_angless[1]["TorsoRFemur_angle"],
                              "LTibiaSelf_vector": df_angless[1]["LTibiaSelf_vector"],
                              "RTibiaSelf_vector": df_angless[1]["RTibiaSelf_vector"]})
    df_angles["Time_in_sec"] = [n / fps for n in range(len(df_angles))]
    if show_plot_angle_demo:
        plot_angles("CAM[Fixed]", pd.DataFrame(df_angles))

    # 分析步态周期
    Gait_Analysis.analysis(df_angles=pd.DataFrame(df_angles), fps=fps, pts_cam=pts_cams_ndarray[1], analysis_keypoint=PoseLandmark.RIGHT_KNEE)

    plt.show()
