import json
import sys

import cv2 as cv
import mediapipe as mp
import numpy as np
import argparse

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

poseDetectorPool = []
frame_shape = [720, 1280]

checked_pose_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                          26, 27, 28, 29, 30, 31, 32]


def BGR(RGB: tuple):
    return RGB[2], RGB[1], RGB[0]


def drawHealBoneLogo(*frames):
    for index, frame in enumerate(*frames):
        logo = cv.imread('./logo.png')
        width = 123 * 2
        height = int(width / 4.3)
        logo = cv.resize(logo, (width, height))
        img2gray = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(img2gray, 1, 255, cv.THRESH_BINARY)
        roi = frame[-height - 10:-10, -width - 10:-10]
        roi[np.where(mask)] = 0
        roi += logo


def read_video_frames(*streams: str, callback):
    """
    从视频流中读取帧，并将帧传递给回调函数
    :param streams:
    :param callback:
    """
    caps = [cv.VideoCapture(stream) for stream in streams]
    pts_cams = [[] for i in range(len(caps))]
    pts_3d = []

    print("read_video_frames:", len(caps))

    for cap in caps:
        # 视频流的分辨率设置为1280x720
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    while True:
        frames = []
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

        pose_landmarks, pose_world_landmarks, pose_landmarks_proto, pose_world_landmarks_proto = callback(*frames)

        # 解除写入性能限制，将RGB转换为BGR
        for index, frame in enumerate(frames):
            frames[index].flags.writeable = True
            frames[index] = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # 将归一化的坐标转换为原始坐标
        for index, pose_landmark in enumerate(pose_landmarks):
            if pose_landmark:
                for keypoint_index, landmark in enumerate(pose_landmark):
                    # 只处理待检测的关键点，用于后续CheckCube扩展
                    if keypoint_index not in checked_pose_keypoints:
                        continue
                    truth_x = int(round(landmark.x * frames[index].shape[1]))
                    truth_y = int(round(landmark.y * frames[index].shape[0]))
                    cv.circle(frames[index], (truth_x, truth_y), 3, BGR(RGB=(255, 0, 0)), -1)
                    pts_cams[index].append([truth_x, truth_y])
            else:
                pts_cams[index] = [[-1, -1]] * len(checked_pose_keypoints)

        for index, pose_landmark_proto in enumerate(pose_landmarks_proto):
            mp_drawing.draw_landmarks(frames[index], pose_landmark_proto, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # 绘制HealBone图标
        drawHealBoneLogo(frames)

        # 窗口展示视频帧
        for index, frame in enumerate(frames):
            cv.imshow("cam" + str(index), frame)

        k = cv.waitKey(1)
        # 按ESC键退出
        if k & 0xFF == 27:
            break

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    return [np.array(pts_cam) for pts_cam in pts_cams], np.array(pts_3d)


def inferPose(*video_frames):
    """
    推断视频帧中人体姿态
    :param video_frames:
    """
    global poseDetectorPool
    if len(poseDetectorPool) == 0:
        poseDetectorPool = [mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True,
            smooth_segmentation=True,
        ) for i in range(len(video_frames))]
        print("PoseDetectorPool Size:", len(poseDetectorPool))

    results = []
    for i in range(len(video_frames)):
        results.append(poseDetectorPool[i].process(video_frames[i]))
    return results


def video_frame_handler(*video_frame):
    infer_results = inferPose(*video_frame)

    pose_landmarks = [i.pose_landmarks.landmark for i in infer_results]
    pose_world_landmarks = [i.pose_world_landmarks.landmark for i in infer_results]

    pose_landmarks_proto = [i.pose_landmarks for i in infer_results]
    pose_world_landmarks_proto = [i.pose_world_landmarks for i in infer_results]

    return pose_landmarks, pose_world_landmarks, pose_landmarks_proto, pose_world_landmarks_proto


def save_keypoints(filename, pts):
    file = open(filename, "w")
    json.dump(pts.tolist(), file)
    file.close()


if __name__ == '__main__':
    # 预先读取的不同视角视频
    input_stream1 = 'data/view-side.mp4'
    input_stream2 = 'data/view-front.mp4'
    show_mediapipe_drawing = True

    # 读取相机串口编号
    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    # opencv读取视频source，并使用mediapipe进行KeyPoints推理
    pts_cams, pts_3d = read_video_frames(input_stream1, input_stream2,
                                         callback=lambda frame0, frame1: video_frame_handler(frame0, frame1))

    # 保存原始的推理结果
    for index, pts_cam in enumerate(pts_cams):
        save_keypoints('pts_cam' + str(index) + '.json', pts_cam)

    # 保存fixed过后的3D空间推理结果
    save_keypoints('pts_3d.json', pts_3d)
