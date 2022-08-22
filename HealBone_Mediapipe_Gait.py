import json
import sys
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

from acceleration import sensormotionDemo

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

poseDetectorPool = []
frame_shape = [720, 1280]

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

    fig, axes = plt.subplots(3, 3, figsize=(24, 14))

    fig.suptitle("关节角度变化周期 - " + title)

    sns.lineplot(ax=axes[0, 0], data=df, x="Time_in_sec", y="TorsoLHip_angle").set(xlabel="时间（秒）",
                                                                                   ylabel="躯干 L 髋关节角度 (°)")
    sns.lineplot(ax=axes[0, 1], data=df, x="Time_in_sec", y="TorsoRHip_angle").set(xlabel="时间（秒）",
                                                                                   ylabel="躯干 R 髋关节角度 (°)")
    sns.lineplot(ax=axes[0, 2], data=df, x="Time_in_sec", y="LHip_angle").set(xlabel="时间（秒）",
                                                                              ylabel="L 髋关节角度 (°)")
    sns.lineplot(ax=axes[1, 0], data=df, x="Time_in_sec", y="RHip_angle").set(xlabel="时间（秒）",
                                                                              ylabel="R 髋关节角度 (°)")
    sns.lineplot(ax=axes[1, 1], data=df, x="Time_in_sec", y="LKnee_angle").set(xlabel="时间（秒）",
                                                                               ylabel="L 膝关节角度 (°)")
    sns.lineplot(ax=axes[1, 2], data=df, x="Time_in_sec", y="RKnee_angle").set(xlabel="时间（秒）",
                                                                               ylabel="R 膝关节角度 (°)")
    sns.lineplot(ax=axes[2, 0], data=df, x="Time_in_sec", y="LAnkle_angle").set(xlabel="时间（秒）",
                                                                                ylabel="L 踝关节角度 (°)")
    sns.lineplot(ax=axes[2, 1], data=df, x="Time_in_sec", y="RAnkle_angle").set(xlabel="时间（秒）",
                                                                                ylabel="R 踝关节角度 (°)")

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
    return theta


def build_vector(landmarks, landmark_index: PoseLandmark) -> ndarray:
    """
    根据关节点坐标构建向量
    :param landmark_index:
    :param landmarks:
    :return:
    """
    return np.array(
        [landmarks[landmark_index.value].x, landmarks[landmark_index.value].y, landmarks[landmark_index.value].z])


def landmark_to_angle(landmarks) -> dict:
    """
    计算单次姿态的所有点的检测夹角
    :param landmarks:
    :return:
    """
    # 鼻部坐标
    Nose_coor = build_vector(landmarks, mp_pose.PoseLandmark.NOSE)
    # 左髋关节坐标
    LHip_coor = build_vector(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    # 右髋关节坐标
    RHip_coor = build_vector(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
    # 左右髋关节中点
    MidHip_coor = np.array(
        [(LHip_coor[0] + RHip_coor[0]) / 2, (LHip_coor[1] + RHip_coor[1]) / 2, (LHip_coor[1] + RHip_coor[2]) / 2])
    # 左膝关节坐标
    LKnee_coor = build_vector(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
    # 右膝关节坐标
    RKnee_coor = build_vector(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
    # 左踝关节坐标
    LAnkle_coor = build_vector(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
    # 右踝关节坐标
    RAnkle_coor = build_vector(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
    # 左脚拇指坐标
    LBigToe_coor = build_vector(landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
    # 右脚拇指坐标
    RBigToe_coor = build_vector(landmarks, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)

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
    # 左足部向量
    LFoot_vector = LBigToe_coor - LAnkle_coor
    # 右足部向量
    RFoot_vector = RBigToe_coor - RAnkle_coor

    # 躯干与胯骨的夹角
    TorsoLHip_angle = vectors_to_angle(Torso_vector, Hip_vector)
    TorsoRHip_angle = vectors_to_angle(Torso_vector, Hip_vector)

    # 左股骨与胯骨的夹角
    LHip_angle = vectors_to_angle(LFemur_vector, Hip_vector)
    # 右股骨与胯骨的夹角
    RHip_angle = vectors_to_angle(RFemur_vector, -Hip_vector)

    # 左胫骨与左股骨的夹角
    LKnee_angle = vectors_to_angle(LTibia_vector, LFemur_vector)
    # 右胫骨与右股骨的夹角
    RKnee_angle = vectors_to_angle(RTibia_vector, RFemur_vector)
    # 左踝与左胫骨的夹角
    LAnkle_angle = vectors_to_angle(LFoot_vector, LTibia_vector)
    # 右踝与右胫骨的夹角
    RAnkle_angle = vectors_to_angle(RFoot_vector, RTibia_vector)

    dict_angles = {"TorsoLHip_angle": TorsoLHip_angle, "TorsoRHip_angle": TorsoRHip_angle, "LHip_angle": LHip_angle,
                   "RHip_angle": RHip_angle, "LKnee_angle": LKnee_angle, "RKnee_angle": RKnee_angle,
                   "LAnkle_angle": LAnkle_angle, "RAnkle_angle": RAnkle_angle}
    return dict_angles


def read_video_frames(*streams: str, videoFrameHandler: Callable[[tuple], tuple],
                      computedAnglesCallback: Callable) -> tuple:
    """
    从视频流中读取帧，并将帧传递给回调函数
    :param computedAnglesCallback:
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
    chart_datas: list = [[] for _ in range(len(caps))]

    print("read_video_sources:", len(caps))
    print("fps:", fps[0])

    for cap in caps:
        # 视频流的分辨率设置为1280x720
        cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_shape[1])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_shape[0])

    while True:
        frames: List[ndarray] = []
        # 遍历caps
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != 720:
                frame = frame[:, frame_shape[1] // 2 - frame_shape[0] // 2:frame_shape[1] // 2 + frame_shape[0] // 2]

            # 将BGR转换为RGB
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # 提升性能，不写入内存
            frame.flags.writeable = False
            frames.append(frame)
        if len(frames) is not len(caps):
            print("Error: not all frames read, just {}/{}".format(len(frames), len(caps)))
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

            # 计算每一帧的3D坐标中的角度
            angle_dict = landmark_to_angle(pose_keypoints)
            chart_datas[pose_landmark_index].append(angle_dict)

            if computedAnglesCallback:
                computedAnglesCallback(angle_dict)

        for pose_landmark_proto_index, pose_landmark_proto in enumerate(pose_landmarks_proto):
            mp_drawing.draw_landmarks(frames[pose_landmark_proto_index], pose_landmark_proto, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # 绘制HealBone图标
        draw_healbone_logo(frames)

        # 窗口展示视频帧
        for frame_index, frame in enumerate(frames):
            cv.imshow("HealBone-Mediapipe-Gait: Camera" + str(frame_index), frame)

        k = cv.waitKey(1)
        # 按ESC键退出
        if k & 0xFF == 27:
            break

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    return [np.array(_pts_cam) for _pts_cam in pts_cams], np.array(pts_3d), fps[0], chart_datas


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


def real_time_computed_angles_handler(angle_dict: dict):
    """
    实时计算角度回调函数
    :param angle_dict:
    :return:
    """
    print(angle_dict)


def save_keypoints(filename: str, pts: ndarray) -> NoReturn:
    file = open(filename, "w")
    json.dump(pts.tolist(), file)
    file.close()


if __name__ == '__main__':
    # 预先读取的不同视角视频
    input_stream1 = 'data/exercise-side.mp4'
    input_stream2 = 'data/exercise-front.mp4'
    show_mediapipe_drawing = True

    # 读取相机串口编号
    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    # opencv读取视频source，并使用mediapipe进行KeyPoints推理
    pts_cams_ndarray, pts_3d_ndarray, fps, chart_datas = read_video_frames(input_stream1, input_stream2,
                                                                           videoFrameHandler=lambda frame0, frame1:
                                                                           video_frame_handler(frame0, frame1),
                                                                           computedAnglesCallback=lambda angle_dict:
                                                                           real_time_computed_angles_handler(
                                                                               angle_dict))

    # 绘制步态周期图表
    for chart_index, chart_data in enumerate(chart_datas):
        df_angles = pd.DataFrame(chart_data)
        df_angles["Time_in_sec"] = [n / fps for n in range(len(df_angles))]
        plot_angles("CAM[" + str(chart_index) + "]", df_angles)

    # 保存原始的推理结果，以index为0的推理结果进行3D空间下加速度分解分析
    for index, pts_cam in enumerate(pts_cams_ndarray):
        save_keypoints('pts_cam' + str(index) + '.json', pts_cam)
        if index == 0:
            sensormotionDemo(pts_cam=pts_cam, analysis_keypoint=PoseLandmark.RIGHT_KNEE, fps=fps)

    # 保存fixed过后的3D空间推理结果
    save_keypoints('pts_3d.json', pts_3d_ndarray)
