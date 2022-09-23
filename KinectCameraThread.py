import time
import traceback
from typing import List, Tuple, NoReturn, Any

import numpy as np
from PIL import ImageFont
from PySide2.QtCore import QThread, QMutex
from PySide2.QtGui import QPixmap
from mediapipe.python.solutions.pose import PoseLandmark
from numpy import ndarray
from pyk4a import PyK4A, Config, ColorResolution, FPS, DepthMode, K4AException, ImageFormat
import cv2 as cv
import mediapipe as mp

from GUISignal import VideoFramesSignal, KeyPointsSignal, AngleDictSignal, LogSignal, DetectInterruptSignal, \
    DetectFinishSignal, DetectExitSignal, \
    KinectErrorSignal, PatientTipsSignal, FPSSignal, DistanceSignal
from decorator import FpsPerformance
from evaluate.dsl import *
from kinect_helpers import obj2json, depthInMeters, color_depth_image, colorize

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

"""
待检测的点
"""
KEYPOINT_DETECTED = [
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


class KinectCaptureThread(QThread):

    def __init__(self, k4aConfig: dict, mpConfig: dict, extraConfig: dict, EvaluateMetadata: dict):
        super(KinectCaptureThread, self).__init__()
        self.signal_frames: VideoFramesSignal = VideoFramesSignal()
        self.signal_keypoints: KeyPointsSignal = KeyPointsSignal()
        self.signal_angles: AngleDictSignal = AngleDictSignal()
        self.signal_log: LogSignal = LogSignal()
        self.signal_detectInterrupt = DetectInterruptSignal()
        self.signal_detectFinish = DetectFinishSignal()
        self.signal_detectExit = DetectExitSignal()
        self.signal_kinectError = KinectErrorSignal()
        self.signal_fpsSignal = FPSSignal()
        self.signal_patientTips = PatientTipsSignal()
        self.signal_distance = DistanceSignal()

        self.k4aConfig = k4aConfig
        self.mpConfig = mpConfig

        self.k4a: PyK4A = PyK4A(Config(
            color_resolution=ColorResolution(k4aConfig["color_resolution"]),
            camera_fps=FPS(k4aConfig["camera_fps"]),
            depth_mode=DepthMode(k4aConfig["depth_mode"]),
            synchronized_images_only=k4aConfig["synchronized_images_only"],
            color_format=ImageFormat.COLOR_BGRA32
        ))
        self.mpConfig = mpConfig
        self.poseDetector = mp_pose.Pose(
            min_detection_confidence=mpConfig["min_detection_confidence"],
            min_tracking_confidence=mpConfig["min_tracking_confidence"],
            model_complexity=mpConfig["model_complexity"],
            smooth_landmarks=mpConfig["smooth_landmarks"]
        )
        self.extraConfig = extraConfig
        self.evaluateMetadata = EvaluateMetadata
        self.venv = {}
        self.generateVenv(self.evaluateMetadata["requireCollect"])
        """
        recordFlag控制线程停止
        """
        self.recordFlag = True
        """
        detectFlag识别标识
        """
        self.detectFlag = False
        """
        detect_frames已经识别的视频帧
        """
        self.detect_frames: List = []
        self.emitLog(str(obj2json(k4aConfig)))
        self.mutex = QMutex()
        self.detectStartTime = time.time()
        self.logo = cv.imread('./logo.png')

    def stopCapture(self):
        self.recordFlag = False
        if self.k4a.is_running:
            self.k4a.stop()

    def generateVenv(self, requireCollect):
        for infoItem in requireCollect:
            self.venv[f"${infoItem.name}"] = self.extraConfig[infoItem]
        print("venv", self.venv)

    def emitLog(self, logStr: str):
        self.signal_log.signal.emit(logStr)

    def emitVideoFrames(self, frames: List[ndarray]):
        self.signal_frames.signal.emit(frames)

    def emitKeyPoints(self, frames: List[List]):
        self.signal_keypoints.signal.emit(frames)

    def emitAngles(self, angleDict):
        self.signal_angles.signal.emit(angleDict)

    def emitDetectInterrupt(self, empty="empty"):
        self.signal_detectInterrupt.signal.emit(empty)

    def emitDetectFinish(self, empty="empty"):
        self.signal_detectFinish.signal.emit(empty)

    def emitDetectExit(self, empty="empty"):
        self.signal_detectExit.signal.emit(empty)

    def emitKinectError(self, empty="empty"):
        self.signal_kinectError.signal.emit(empty)

    def emitFPS(self, fps: str):
        self.signal_fpsSignal.signal.emit("FPS " + fps)

    def emitPatientTips(self, tips):
        self.signal_patientTips.signal.emit(tips)

    def emitDistance(self, distance):
        self.signal_distance.signal.emit(distance)

    @staticmethod
    def BGR(RGB: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        RGB Color to BGR Color
        :param RGB: RGB Color
        :return: BGR Color
        """
        return RGB[2], RGB[1], RGB[0]

    def drawHealboneLogo(self, frame: ndarray) -> NoReturn:
        """
        add HealBone Logo to CV-Frame
        """
        width = 123 * 2
        height = int(width / 4.3)
        logo = cv.resize(self.logo, (width, height))
        img2gray = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(img2gray, 1, 255, cv.THRESH_BINARY)
        roi = frame[-height - 10:-10, -width - 10:-10]
        roi[np.where(mask)] = 0
        roi += logo

    # @FpsPerformance
    def processFrames(self, pose_landmarks_proto, rgb_frame, deep_frame):
        """
        多source姿态关键点proto回调函数
        :param pose_landmarks_proto:
        :param rgb_frame:
        :param deep_frame:
        """
        # TODO: addWeighted和colorize性能瓶颈，但非关键因素
        patient_frame = rgb_frame
        combined_image = cv.addWeighted(rgb_frame, 0.5, colorize(deep_frame), 0.5, 0)
        deep_image = colorize(deep_frame, (None, 5000), cv.COLORMAP_HSV)
        if pose_landmarks_proto is not None:
            mp_drawing.draw_landmarks(combined_image, pose_landmarks_proto, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            mp_drawing.draw_landmarks(rgb_frame, pose_landmarks_proto, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            mp_drawing.draw_landmarks(deep_image, pose_landmarks_proto, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # 绘制HealBone图标
        self.drawHealboneLogo(combined_image)
        self.drawHealboneLogo(rgb_frame)
        self.drawHealboneLogo(patient_frame)

        return [
            cv.cvtColor(cv.resize(rgb_frame, (0, 0), fx=0.6, fy=0.6), cv.COLOR_BGR2RGB),
            cv.cvtColor(cv.resize(deep_image, (0, 0), fx=0.6, fy=0.6), cv.COLOR_BGR2RGB),
            cv.cvtColor(cv.resize(combined_image, (0, 0), fx=0.6, fy=0.6), cv.COLOR_BGR2RGB),
            cv.cvtColor(cv.resize(patient_frame, (0, 0), fx=0.6, fy=0.6), cv.COLOR_BGR2RGB),
        ]

    def run(self):
        try:
            self.k4a.start()
            self.emitLog("Kinect配置: " + str(obj2json(self.k4aConfig)))
            self.emitLog("姿势估计器配置: " + str(obj2json(self.mpConfig)))
            self.emitLog("ExtraDetectionDimension: " + str(self.extraConfig))
            self.emitLog("检测模式: " + str(self.evaluateMetadata))
            self.emitLog("等待目标进入检测范围...")
            while True:
                start_time = time.time()
                if not self.k4a.opened:
                    break
                capture = self.k4a.get_capture()

                """
                原始的RGBA视频帧
                """
                frame = capture.color[:, :, :3]

                depth_image_raw = capture.transformed_depth

                """
                # OpenCV自带的去噪修复，帧率太低
                # depth_image_raw = smooth_depth_image(depth_image_raw, max_hole_size=10)

                # 孔洞填充滤波器
                # hole_filter = HoleFilling_Filter(flag='min')
                # depth_image_raw = hole_filter.smooth_image(depth_image_raw)
                # 去噪滤波器
                # noise_filter = Denoising_Filter(flag='modeling', theta=60)
                # depth_image_raw = noise_filter.smooth_image(depth_image_raw)
                """

                # 深度图像数据归一化为米
                depth_image = depthInMeters(depth_image_raw)

                if np.any(capture.depth):
                    # 将BGR转换为RGB
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    # frame_gpu = cv.cuda_GpuMat()
                    # frame_gpu.upload(frame)
                    # frame = cv.cuda.cvtColor(frame_gpu, cv.COLOR_BGR2RGB).download()

                    # 提升性能，不写入内存
                    frame.flags.writeable = False
                    pose_landmarks, pose_world_landmarks, pose_landmarks_proto, pose_world_landmarks_proto = self.videoFrameHandler(
                        frame)
                    frame.flags.writeable = True

                    # frame_gpu.upload(frame)
                    # frame = cv.cuda.cvtColor(frame_gpu, cv.COLOR_RGB2BGR).download()

                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

                    """
                    ！！判断评估是否结束！！
                    """
                    self.venv["$detectStartTime"] = self.detectStartTime
                    if not self.recordFlag or DSL(self.evaluateMetadata["calcRules"]["end"], self.venv):
                        if self.detectFlag:
                            self.emitDetectFinish()
                            self.emitLog("检测结束")
                            self.emitPatientTips("检测完成，等待生成报告")
                            self.k4a.stop()
                            break

                    if pose_landmarks is None or pose_world_landmarks is None or pose_landmarks_proto is None or \
                            pose_world_landmarks_proto is None:
                        self.emitPatientTips("请进入相机范围")
                        self.emitVideoFrames(
                            self.processFrames(pose_landmarks_proto, frame, color_depth_image(depth_image_raw)))
                        self.emitFPS(str(round(1 / (time.time() - start_time))))
                        if self.detectFlag:
                            self.detectStartTime = time.time()
                            self.detectFlag = False
                            self.detect_frames = []
                            self.emitLog("检测过程中断！等待重新检测")
                            print("检测过程被中断！等待重新检测")
                        continue
                    # 将归一化的坐标转换为原始坐标
                    pose_keypoints = []

                    for pose_landmark_index, pose_landmark in enumerate(pose_landmarks):
                        if pose_landmark:
                            # 只处理待检测的关键点，用于后续CheckCube扩展
                            if PoseLandmark(pose_landmark_index) not in KEYPOINT_DETECTED:
                                continue
                            visualize_x = int(round(pose_landmark.x * frame.shape[1]))
                            visualize_y = int(round(pose_landmark.y * frame.shape[0]))
                            truth_x = pose_landmark.x
                            truth_y = pose_landmark.y
                            # MediaPipe原始的landmark_z不可信
                            discard_z = pose_landmark.z
                            deep_axis1 = visualize_y if visualize_y < depth_image.shape[0] else depth_image.shape[0] - 1
                            deep_axis2 = visualize_x if visualize_x < depth_image.shape[1] else depth_image.shape[1] - 1
                            deep_z = depth_image[deep_axis1 if deep_axis1 > 0 else 0,
                                                 deep_axis2 if deep_axis2 > 0 else 0]
                            visibility = pose_landmark.visibility
                            cv.putText(frame, "Depth:" + str(
                                round(deep_z, 3)),
                                       (visualize_x - 10, visualize_y - 10),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.5, self.BGR(RGB=(102, 153, 250)), 1,
                                       cv.LINE_AA)
                            # cv.circle(frame, (visualize_x, visualize_y), radius=3, color=self.BGR(RGB=(255, 0, 0)), thickness=-1)
                            pose_keypoints.append([truth_x, truth_y, deep_z, visibility])
                        else:
                            pose_keypoints = [[-1, -1, -1, -1]] * len(KEYPOINT_DETECTED)

                    """
                    ！！判断是否达到评估开始标准！！
                    """
                    self.venv["$keypoints"] = pose_keypoints
                    if DSL(self.evaluateMetadata["calcRules"]["start"], self.venv):
                        if not self.detectFlag:
                            self.emitLog("已识别到所有检测点，开始检测")
                            self.emitPatientTips("已识别到所有检测点，开始检测，请开始行走")
                            self.emitDetectInterrupt()
                            self.detectStartTime = time.time()
                            print("开始检测！")
                            self.detectFlag = True
                        self.detect_frames.append(pose_keypoints)
                        self.emitPatientTips(
                            "已检测" + str(round(time.time() - self.detectStartTime, 1)) + '秒，剩余' + str(
                                round(self.venv["$time"] - (time.time() - self.detectStartTime),
                                      1)) + '秒')
                        self.emitLog("已检测" + str(time.time() - self.detectStartTime) + '秒')
                        self.emitKeyPoints(pose_keypoints)
                        self.generateVenvVectors(pose_keypoints)
                        self.emitAngles(self.calculateAnglesMediaPipe(pose_keypoints))
                    else:
                        self.emitPatientTips("调整姿势使身体和四肢完全包含在相机视图中")
                        if self.detectFlag:
                            self.detectFlag = False
                            self.detect_frames = []
                            self.emitLog("检测过程被打断！等待重新检测")
                            self.emitPatientTips("检测过程被打断！重新调整姿势，并等待重新检测")
                            print("检测过程被打断！等待重新检测")
                            continue
                    self.emitDistance((pose_keypoints[23][2] + pose_keypoints[24][2] + pose_keypoints[11][2] +
                                       pose_keypoints[12][2]) / 4)
                    self.emitVideoFrames(
                        self.processFrames(pose_landmarks_proto, frame, color_depth_image(depth_image_raw)))

                del capture
                self.emitFPS(str(round(1 / (time.time() - start_time))))

        except K4AException as e:
            self.emitLog(repr(e))
            self.emitLog(traceback.format_exc())
            self.emitLog("Kinect设备打开失败, 请检查Kinect是否被其他进程占用")
            self.emitKinectError()
        """
        释放MediaPipe姿势估计器
        """
        self.emitDetectExit()
        self.poseDetector.close()
        if self.k4a.opened:
            self.k4a.stop()
        self.quit()

    def performance(self, start_time, TAG=""):
        __time__ = (time.time() - start_time)
        try:
            print(f"{TAG} run time {__time__ * 1000}ms, FPS {round(1 / __time__, 2)}")
        except ZeroDivisionError as e:
            print(f"{TAG} run time {__time__ * 1000}ms, FPS MAX")

    # @FpsPerformance
    def videoFrameHandler(self, video_frame):
        """
        每一帧视频帧被读取到时进行姿势推理的异步Handler
        :param video_frame:
        :returns: pose_landmarks, pose_world_landmarks, pose_landmarks_proto, pose_world_landmarks_proto
        """
        infer_result = self.inferPose(video_frame)

        if infer_result.pose_landmarks is None or infer_result.pose_world_landmarks is None:
            return None, None, None, None
        pose_landmarks = infer_result.pose_landmarks.landmark
        pose_world_landmarks = infer_result.pose_world_landmarks.landmark

        pose_landmarks_proto = infer_result.pose_landmarks
        pose_world_landmarks_proto = infer_result.pose_world_landmarks

        return pose_landmarks, pose_world_landmarks, pose_landmarks_proto, pose_world_landmarks_proto

    # @FpsPerformance
    def inferPose(self, video_frame) -> Any:
        """
        推断视频帧中人体姿态
        :param video_frame:
        :return
        """
        return self.poseDetector.process(video_frame)

    @staticmethod
    def buildVector(keypoints, keypoint_index) -> ndarray:
        """
        根据关节点坐标构建向量
        :param keypoints:
        :param keypoint_index:
        :return:
        """
        return np.array(
            [keypoints[keypoint_index][0], keypoints[keypoint_index][1], keypoints[keypoint_index][2]])

    @staticmethod
    def vectors_to_angle(vector1, vector2, supplementaryAngle=False) -> float:
        """
        计算两个向量之间的夹角
        :param supplementaryAngle: 
        :param vector1:
        :param vector2:
        :return:
        """
        x = np.dot(vector1, -vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        theta = np.degrees(np.arccos(x))
        return theta if not supplementaryAngle else 180 - theta

    def generateVenvVectors(self, keypoints):
        for enumItem in mp_pose.PoseLandmark:
            self.venv[f"$k{enumItem.value}"] = self.buildVector(keypoints, enumItem)
        self.venv[f"$torso"] = (self.venv[f"$k23"] + self.venv[f"$k24"]) / 2 - (self.venv[f"$k11"] + self.venv[f"$k12"]) / 2
        self.venv[f"$L$femur"] = self.venv[f"$26"] - self.venv[f"$24"]
        self.venv[f"$R$femur"] = self.venv[f"$25"] - self.venv[f"$23"]
        self.venv[f"$L$tibia"] = self.venv[f"$26"] - self.venv[f"$28"]
        self.venv[f"$L$tibia"] = self.venv[f"$25"] - self.venv[f"$27"]


    def calculateAnglesMediaPipe(self, keypoints) -> dict:
        """
        计算单次姿态的所有点的检测夹角
        :param landmarks:
        :return:
        """
        # 鼻部坐标
        Nose_coor = self.buildVector(keypoints, mp_pose.PoseLandmark.NOSE)
        # 左髋关节坐标
        LHip_coor = self.buildVector(keypoints, mp_pose.PoseLandmark.LEFT_HIP)
        # 右髋关节坐标
        RHip_coor = self.buildVector(keypoints, mp_pose.PoseLandmark.RIGHT_HIP)
        # 左右髋关节中点
        MidHip_coor = (LHip_coor + RHip_coor)/2
        # 左膝关节坐标
        LKnee_coor = self.buildVector(keypoints, mp_pose.PoseLandmark.LEFT_KNEE)
        # 右膝关节坐标
        RKnee_coor = self.buildVector(keypoints, mp_pose.PoseLandmark.RIGHT_KNEE)
        # 左踝关节坐标
        LAnkle_coor = self.buildVector(keypoints, mp_pose.PoseLandmark.LEFT_ANKLE)
        # 右踝关节坐标
        RAnkle_coor = self.buildVector(keypoints, mp_pose.PoseLandmark.RIGHT_ANKLE)
        # 左脚拇指坐标
        LBigToe_coor = self.buildVector(keypoints, mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
        # 右脚拇指坐标
        RBigToe_coor = self.buildVector(keypoints, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)

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
        TorsoLHip_angle = self.vectors_to_angle(Torso_vector, Hip_vector, supplementaryAngle=True)
        TorsoRHip_angle = self.vectors_to_angle(Torso_vector, -Hip_vector, supplementaryAngle=True)

        # 内收外展
        # 左股骨与胯骨的夹角
        LHip_angle = self.vectors_to_angle(LFemur_vector, Hip_vector, supplementaryAngle=True)
        # 右股骨与胯骨的夹角
        RHip_angle = self.vectors_to_angle(RFemur_vector, -Hip_vector, supplementaryAngle=True)

        # 屈曲伸展
        # 躯干与股骨
        TorsoLFemur_angle = self.vectors_to_angle(Torso_vector, LFemur_vector, supplementaryAngle=True)
        TorsoRFemur_angle = self.vectors_to_angle(Torso_vector, RFemur_vector, supplementaryAngle=True)

        # 外旋内旋
        # 胫骨旋转
        LTibiaSelf_vector = self.vectors_to_angle(LTibia_vector, np.array([0, 1, 0]), supplementaryAngle=True)
        RTibiaSelf_vector = self.vectors_to_angle(RTibia_vector, np.array([0, 1, 0]), supplementaryAngle=True)

        # 左胫骨与左股骨的夹角
        LKnee_angle = self.vectors_to_angle(LTibia_vector, LFemur_vector, supplementaryAngle=True)
        # 右胫骨与右股骨的夹角
        RKnee_angle = self.vectors_to_angle(RTibia_vector, RFemur_vector, supplementaryAngle=True)
        # 左踝与左胫骨的夹角
        LAnkle_angle = self.vectors_to_angle(LFoot_vector, LTibia_vector, supplementaryAngle=False)
        # 右踝与右胫骨的夹角
        RAnkle_angle = self.vectors_to_angle(RFoot_vector, RTibia_vector, supplementaryAngle=False)

        dict_angles = {"frame_index": len(self.detect_frames), "Time_in_sec": (time.time() - self.detectStartTime),
                       "TorsoLHip_angle": TorsoLHip_angle, "TorsoRHip_angle": TorsoRHip_angle, "LHip_angle": LHip_angle,
                       "RHip_angle": RHip_angle, "LKnee_angle": LKnee_angle, "RKnee_angle": RKnee_angle,
                       "TorsoLFemur_angle": TorsoLFemur_angle, "TorsoRFemur_angle": TorsoRFemur_angle,
                       "LTibiaSelf_vector": LTibiaSelf_vector, "RTibiaSelf_vector": RTibiaSelf_vector,
                       "LAnkle_angle": LAnkle_angle, "RAnkle_angle": RAnkle_angle}
        return dict_angles
