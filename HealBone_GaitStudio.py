import sys
import time
from typing import List

import cv2
import pandas as pd
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import qimage2ndarray
import pyqtgraph as pg
from mediapipe.python.solutions.pose import PoseLandmark
from qtmodernredux import QtModernRedux

import Gait_Analysis_GUI
import MainWindow
from GUISignal import LogSignal
from KinectCameraThread import KinectCaptureThread

from decorator import FpsPerformance
from evaluate.QRequireCollectDialog import QRequireCollectDialog
from evaluate.util import EvaluateMetadata
from widgets.QDataFrameTable import DataFrameTable
from widgets.QMaximumDockWidget import QMaximumDockWidget


class HealBoneWindow(QMainWindow, MainWindow.Ui_MainWindow):
    class HealBoneViewModel(object):
        detectStatus = False
        anglesCheckCube = [
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
        anglesViewerRange = 6
        patientMode = False
        currentPatientTips = ""
        fpsStr = ""
        currentPatientDistance = ""

    def __init__(self, *args):
        QMainWindow.__init__(self, *args)
        MainWindow.Ui_MainWindow.__init__(self)
        self.threadCapture: KinectCaptureThread = None
        self.viewModel = self.HealBoneViewModel()
        """
        无边框窗口
        """
        if use_modern_ui:
            self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setupUi(self)

        """
        合并右侧悬浮窗口
        """
        self.tabifyDockWidget(self.angleViewerDock, self.viewerDock)
        """
        焦点角度窗口
        """
        self.angleViewerDock.raise_()
        """
        角度Dock事件过滤器
        """
        self.angleViewerDock.installEventFilter(self)
        """
        Tab事件过滤器
        """
        self.tabWidget.installEventFilter(self)
        """
        构建QTGraph实时显示角度
        """
        self.anglePltLayouts: List[List[pg.PlotWidget]] = []
        self.anglePltDataList: List[List[List]] = []
        for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
            self.anglePltLayouts.append([])
            self.anglePltDataList.append([])
            for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                label_style = {'color': '#EEE', 'font-size': '12px', 'font-family': '微软雅黑'}

                self.anglePltDataList[anglesCubeIndex].append([[], []])
                self.anglePltLayouts[anglesCubeIndex].append(
                    pg.PlotWidget(parent=self.angleViewerDockScrollAreaContents, background="#2F2F2F"))
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setTitle(anglesCube["title"] + " " + angleCube[1],
                                                                               **label_style)
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setAntialiasing(True)
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setXRange(0, self.viewModel.anglesViewerRange)
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setYRange(anglesCube["ylim"][0],
                                                                                anglesCube["ylim"][1])
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].showGrid(x=True, y=True)

                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setMinimumHeight(180)

                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setLabel('left', angleCube[3], **label_style)
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setLabel('bottom', angleCube[2], **label_style)
                self.angleViewerDockScrollAreaContentsLayout.addWidget(
                    self.anglePltLayouts[anglesCubeIndex][angleCubeIndex])
        """
        Linked X label
        """
        for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
            for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                """
                第一行第一个exclude
                """
                if anglesCubeIndex == 0 and angleCubeIndex == 0:
                    continue
                """
                关联前一行
                """
                # if anglesCubeIndex != 0 and angleCubeIndex == 0:
                #     self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setXLink(self.anglePltLayouts[anglesCubeIndex - 1][0])
                """
                每一行第一个exclude
                """
                if angleCubeIndex == 0:
                    continue
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setXLink(
                    self.anglePltLayouts[anglesCubeIndex][angleCubeIndex - 1])

        self.btnStart.clicked.connect(self.btnStartClicked)
        self.hsAnglesViewerRange.valueChanged.connect(self.changeHsAnglesViewerRange)
        self.hsAnglesViewerItems.valueChanged.connect(self.changeHsAnglesViewerItems)
        self.hsMinTrackingConfidence.valueChanged.connect(self.changeMinTrackingConfidence)
        self.hsMinDetectionConfidence.valueChanged.connect(self.changeMinDetectionConfidence)
        self.cbPlotAlignBtn.stateChanged.connect(self.changeCbPlotAlign)
        self.cbPatientBtn.stateChanged.connect(self.changeCbPatientBtn)
        """
        创建隐藏的患者视图窗口
        """
        self.cameraPatientView = QGraphicsView(self.KinectIRFOV)
        self.cameraPatientView.setObjectName(u"cameraPatientView")
        self.cameraPatientView.setGeometry(QRect(0, 0, 741, 515))
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cameraPatientView.sizePolicy().hasHeightForWidth())
        self.cameraPatientView.setSizePolicy(sizePolicy)
        self.patientWin = QMaximumDockWidget(self)
        self.patientWin.setWindowTitle("PatientView")
        self.patientWin.setWidget(self.cameraPatientView)

        self.patientWin.setFeatures(QDockWidget.DockWidgetFloatable)
        self.patientWin.setAllowedAreas(Qt.NoDockWidgetArea)

        self.patientWin.setFloating(True)
        self.patientWin.setHidden(True)
        self.patientWin.setMinimumSize(QSize(741, 515))
        self.showStatusMessage("Info: Healbone GaitStudio组件加载完毕")
        self.anglesDataFrame = pd.DataFrame()
        self.dockWidgetContentsLayout = QVBoxLayout(self.anglesDockWidgetContents)
        self.anglesDataFrameTable = DataFrameTable(self.anglesDataFrame)
        self.dockWidgetContentsLayout.addWidget(self.anglesDataFrameTable)
        self.hideLogoFrame = True
        self.pts_cams = []
        for detectIndex, detectionItem in enumerate(EvaluateMetadata):
            self.cbPosturalAssessment.addItem(detectionItem["name"])

    def showStatusMessage(self, text, timeout=2000):
        self.statusBar.showMessage(text, timeout)

    def changeCbPatientBtn(self):
        self.viewModel.patientMode = self.cbPatientBtn.isChecked()
        if self.viewModel.patientMode and self.viewModel.detectStatus:
            self.patientWin.setHidden(False)
        else:
            self.patientWin.setHidden(True)

    def changeCbPlotAlign(self):
        for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
            for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                if self.cbPlotAlignBtn.isChecked():
                    self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].showButtons()
                else:
                    self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].hideButtons()

    def changeHsAnglesViewerRange(self):
        self.tvAnglesRangeFrames.setText("Angles-Viewer Range (" + str(self.hsAnglesViewerRange.value()) + " sec)")
        self.viewModel.anglesViewerRange = int(self.hsAnglesViewerRange.value())
        for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
            for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                label_style = {'color': '#EEE', 'font-size': '12px', 'font-family': '微软雅黑'}
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setXRange(0, self.viewModel.anglesViewerRange)

    def changeMinTrackingConfidence(self):
        self.tvMinTrackingConfidence.setText(
            "Min Tracking Confidence (" + str(round(self.hsMinTrackingConfidence.value() / 100, 1)) + ")")

    def changeMinDetectionConfidence(self):
        self.tvMinDetectionConfidence.setText(
            "Min Detection Confidence (" + str(round(self.hsMinDetectionConfidence.value() / 100, 1)) + ")")

    def changeHsAnglesViewerItems(self):
        for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
            for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                self.tvAnglesItemVisibleNum.setText(
                    "Angle-Items Visible Num (" + str(int(self.hsAnglesViewerItems.value() / 10)) + " Items)")
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setMinimumHeight(
                    int(self.angleViewerDock.size().height() / int(self.hsAnglesViewerItems.value() / 10) - 20))

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Resize and watched == self.tabWidget:
            self.resizeCameraView()
        if event.type() == QEvent.WindowStateChange and watched == self.angleViewerDock:
            if self.angleViewerDock.isFloating() and event.type() == QEvent.Resize and watched == self.angleViewerDock:
                for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
                    for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                        self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setMinimumHeight(
                            int(event.size().height() / 6 - 20))
            if not self.angleViewerDock.isFloating() and event.type() == QEvent.Resize and watched == self.angleViewerDock:
                for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
                    for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                        self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setMinimumHeight(
                            int(event.size().height() / 3 - 20))
        return super().eventFilter(watched, event)

    def resizeCameraView(self, tabWidth=-1, tabHeight=-1):
        tabWidgetSize: QSize = self.tabWidget.geometry().size()
        self.cameraIrFovView.setGeometry(self.cameraIrFovView.x(), self.cameraIrFovView.y(),
                                         tabWidgetSize.width() if tabWidth == -1 else tabWidth,
                                         (tabWidgetSize.height() - 24) if tabHeight == -1 else tabHeight)
        self.cameraFovView.setGeometry(self.cameraFovView.x(), self.cameraFovView.y(),
                                       tabWidgetSize.width() if tabWidth == -1 else tabWidth,
                                       (tabWidgetSize.height() - 24) if tabHeight == -1 else tabHeight)
        self.cameraIrView.setGeometry(self.cameraIrView.x(), self.cameraIrView.y(),
                                      tabWidgetSize.width() if tabWidth == -1 else tabWidth,
                                      (tabWidgetSize.height() - 24) if tabHeight == -1 else tabHeight)
        self.logoFrame.setGeometry(
            self.tabWidget.geometry().x() + tabWidgetSize.width() / 2 - self.logoFrame.width() / 2,
            self.tabWidget.geometry().y() + tabWidgetSize.height() / 2 - self.logoFrame.height() / 2, 481, 191)
        self.logoFrame_2.setGeometry(
            self.tabWidget.geometry().x() + tabWidgetSize.width() / 2 - self.logoFrame.width() / 2,
            self.tabWidget.geometry().y() + tabWidgetSize.height() / 2 - self.logoFrame.height() / 2, 481, 191)
        self.logoFrame_3.setGeometry(
            self.tabWidget.geometry().x() + tabWidgetSize.width() / 2 - self.logoFrame.width() / 2,
            self.tabWidget.geometry().y() + tabWidgetSize.height() / 2 - self.logoFrame.height() / 2, 481, 191)

    def receiveKeyPoints(self, pose_keypoints):
        self.pts_cams.append(pose_keypoints)

    def plotFrameAngles(self, angles: dict):
        self.anglesDataFrame = pd.concat([self.anglesDataFrame, pd.DataFrame([angles])], ignore_index=True)
        self.anglesDataFrameTable.set_data(self.anglesDataFrame)
        for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
            for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                self.anglePltDataList[anglesCubeIndex][angleCubeIndex][0].append(angles["Time_in_sec"])
                self.anglePltDataList[anglesCubeIndex][angleCubeIndex][1].append(angles[angleCube[1]])
                """
                折线图自动滚动
                """
                if self.anglePltDataList[anglesCubeIndex][angleCubeIndex][0][-1] > self.viewModel.anglesViewerRange:
                    self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setXRange(
                        self.anglePltDataList[anglesCubeIndex][angleCubeIndex][0][
                            -1] - self.viewModel.anglesViewerRange,
                        self.anglePltDataList[anglesCubeIndex][angleCubeIndex][0][-1])

                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].getPlotItem().plot(
                    x=self.anglePltDataList[anglesCubeIndex][angleCubeIndex][0],
                    y=self.anglePltDataList[anglesCubeIndex][angleCubeIndex][1],
                    pen=({'color': "r", "width": 1.5}), clear=True)

    def clearAnglesViewer(self):
        self.pts_cams = []
        for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
            for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                self.anglePltDataList[anglesCubeIndex][angleCubeIndex] = [[], []]
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].getPlotItem().plot(
                    self.anglePltDataList[anglesCubeIndex][angleCubeIndex][0],
                    self.anglePltDataList[anglesCubeIndex][angleCubeIndex][1],
                    pen=({'color': "r", "width": 1.5}), clear=True)
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setXRange(0, self.viewModel.anglesViewerRange)
        """
        清空表格
        """
        self.anglesDataFrame = pd.DataFrame()
        self.anglesDataFrameTable.set_data(self.anglesDataFrame)

    def displayCVFrames(self, frames):
        if self.hideLogoFrame:
            self.hideLogoFrame = False
            self.logoFrame.hide()
            self.logoFrame_2.hide()
            self.logoFrame_3.hide()
        self.displayCVFrame(self.cameraIrFovView, frames[2])
        self.displayCVFrame(self.cameraFovView, frames[1])
        self.displayCVFrame(self.cameraIrView, frames[0])
        if self.viewModel.patientMode:
            self.displayCVFrame(self.cameraPatientView, frames[3], patientMode=True)
            self.drawPatientText(self.viewModel.currentPatientTips)

    def displayCVFrame(self, cameraView, frame, patientMode=False):
        """
        将cv的frame显示到label上
        """
        image = qimage2ndarray.array2qimage(frame)  # Solution for memory leak
        jpg_out = QPixmap.fromImage(image).scaled(cameraView.width(), cameraView.height(),
                                                  Qt.KeepAspectRatioByExpanding)
        if cameraView.scene() is None:
            cameraView.setScene(QGraphicsScene())
        qGraphicsPixmapItem = QGraphicsPixmapItem(jpg_out)
        cameraView.scene().clear()
        cameraView.scene().addItem(qGraphicsPixmapItem)  # 将场景添加至视图
        if patientMode:
            layer = QPixmap("./resources/layer.png")
            layerItem = QGraphicsPixmapItem(layer)
            layerItem.setPos(self.patientWin.window().width() / 2 - (layer.width() / 2),
                             self.patientWin.window().height() / 2 - (layer.height() / 2))
            cameraView.scene().addItem(layerItem)
        self.drawFPSText(cameraView, self.viewModel.fpsStr)

    def stopDetect(self):
        self.viewModel.detectStatus = False
        self.threadCapture.stopCapture()
        self.btnStart.setText("开始检测")

    def startDetect(self, k4aConfig, mpConfig, extraConfig, evaluateMetadata):
        self.viewModel.detectStatus = True
        self.btnStart.setText("停止检测")
        self.threadCapture = KinectCaptureThread(k4aConfig=k4aConfig, mpConfig=mpConfig,
                                                 extraConfig=extraConfig, EvaluateMetadata=evaluateMetadata)
        """
        透传子线程中的日志
        """
        self.threadCapture.signal_log.signal.connect(self.logViewAppend)
        """
        透传子线程传出的待显示的视频帧
        """
        self.threadCapture.signal_frames.signal.connect(self.displayCVFrames)
        """
        透传子线程传出的角度dict
        """
        self.threadCapture.signal_angles.signal.connect(self.plotFrameAngles)
        """
        打断时清空角度视图
        """
        self.threadCapture.signal_detectInterrupt.signal.connect(self.clearAnglesViewer)
        """
        透传子线程DetectExit
        """
        self.threadCapture.signal_detectExit.signal.connect(self.detectExit)
        """
        透传子线程DetectFinish
        """
        self.threadCapture.signal_detectFinish.signal.connect(self.detectFinish)
        """
        KinectError
        """
        self.threadCapture.signal_kinectError.signal.connect(
            lambda x: self.showErrorMessage(title="Kinect Error",
                                            content="Kinect设备打开失败, 请检查Kinect是否被其他进程占用"))
        """
        FPSEvent
        """
        self.threadCapture.signal_fpsSignal.signal.connect(self.showFps)
        """
        PatientTips
        """
        self.threadCapture.signal_patientTips.signal.connect(self.showPatientTips)
        """
        signal_keypoints
        """
        self.threadCapture.signal_keypoints.signal.connect(self.receiveKeyPoints)
        """
        Distance
        """
        self.threadCapture.signal_distance.signal.connect(self.showPatientDistance)
        self.threadCapture.start()

    def drawFPSText(self, cameraView, fpsStr):
        textItem = QGraphicsTextItem()
        textItem.setPlainText(fpsStr)
        textItem.setFont(QFont("微软雅黑", 16))
        textItem.setDefaultTextColor(Qt.green)
        textItem.setPos(10, 5)
        if cameraView.scene() is None:
            cameraView.setScene(QGraphicsScene())
        cameraView.scene().addItem(textItem)

    def drawPatientText(self, tips):
        textItem = QGraphicsTextItem()
        textItem.setPlainText(tips)
        # font_size = 36
        font_size = 24
        textItem.setFont(QFont("微软雅黑", font_size))
        textItem.setDefaultTextColor(Qt.red)
        # textItem.setPos(self.patientWin.window().width() / 2 - len(tips) * font_size / 2, self.patientWin.window().height() / 2 - font_size)
        textItem.setPos(10, 30)
        if self.cameraPatientView.scene() is None:
            self.cameraPatientView.setScene(QGraphicsScene())
        self.cameraPatientView.scene().addItem(textItem)

    def showFps(self, fpsStr):
        self.viewModel.fpsStr = fpsStr

    def showPatientTips(self, tips):
        self.viewModel.currentPatientTips = tips

    def showPatientDistance(self, distance):
        self.viewModel.currentPatientDistance = distance

    def showErrorMessage(self, title="Error", content=""):
        self.showStatusMessage(content)
        QMessageBox.critical(self, title, content)

    def showInfoMessage(self, title="Info", content=""):
        self.showStatusMessage(content)
        QMessageBox.information(self, title, content)

    def enableDetectForm(self, enable=True):
        self.cbPosturalAssessment.setEnabled(enable)
        self.cbFPS.setEnabled(enable)
        self.cbDepthMode.setEnabled(enable)
        self.cbColorResolution.setEnabled(enable)
        self.cbModelComplexity.setEnabled(enable)
        self.hsMinTrackingConfidence.setEnabled(enable)
        self.hsMinDetectionConfidence.setEnabled(enable)
        self.cbSmoothLandmarks.setEnabled(enable)

    def detectExit(self):
        if self.viewModel.patientMode:
            self.patientWin.hide()
        self.viewModel.detectStatus = False
        self.enableDetectForm(enable=True)
        self.btnStart.setText("开始检测")
        self.logViewAppend("Pose Detect线程已结束")

    def detectFinish(self):
        """
        检测结束，进入分析报告流程
        :return:
        """
        self.logViewAppend("Pose Detect完成, 结果分析中...")
        self.showInfoMessage(content="Pose Detect完成, 结果分析中...")
        try:
            Gait_Analysis_GUI.analysis(df_angles=pd.DataFrame(self.anglesDataFrame), pts_cam=self.pts_cams,
                                       analysis_keypoint=PoseLandmark.RIGHT_KNEE,
                                       use_modern_ui=use_modern_ui)

            # todo: 1、最大下蹲角度（躯干和） 2、躯干和大腿（侧面） 3、脚踝和小腿（余角） 4、肌肉控制情况
        except AssertionError as e:
            self.logViewAppend(repr(e))
            self.logViewAppend("分析样本数量未达分析标准，请增大检测时长或在规定周期内保证有效动作")
            self.showErrorMessage(content="分析样本数量未达分析标准，请增大检测时长或在规定周期内保证有效动作")

    def btnStartClicked(self):
        if self.viewModel.detectStatus:
            self.stopDetect()
            self.enableDetectForm(enable=True)
            self.patientWin.hide()
        else:
            k4aConfig = {
                "color_resolution": self.cbColorResolution.currentIndex() + 1,
                "camera_fps": self.cbFPS.currentIndex(),
                "depth_mode": self.cbDepthMode.currentIndex() + 1,
                "synchronized_images_only": True
            }
            mpConfig = {
                "min_detection_confidence": round(
                    self.hsMinDetectionConfidence.sliderPosition() / self.hsMinDetectionConfidence.maximum(), 1),
                "min_tracking_confidence": round(
                    self.hsMinTrackingConfidence.sliderPosition() / self.hsMinTrackingConfidence.maximum(), 1),
                "model_complexity": self.cbModelComplexity.currentIndex(),
                "smooth_landmarks": self.cbSmoothLandmarks.isChecked()
            }
            collectDialog = QRequireCollectDialog(
                metadata=EvaluateMetadata[self.cbPosturalAssessment.currentIndex()]["requireCollect"])
            extraConfig = {}
            if collectDialog.exec_() == QDialog.Accepted:
                extraConfig = collectDialog.getResult()
                print(extraConfig)
            else:
                return
            self.clearAnglesViewer()
            self.startDetect(k4aConfig, mpConfig, extraConfig,
                             EvaluateMetadata[self.cbPosturalAssessment.currentIndex()])
            if self.viewModel.patientMode:
                self.patientWin.show()
            self.enableDetectForm(enable=False)

    def logViewAppend(self, text):
        self.outputText.moveCursor(QTextCursor.End, QTextCursor.MoveMode.MoveAnchor)
        local_time_asctimes = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(time.time()))
        self.outputText.setMarkdown(
            self.outputText.toMarkdown(
                QTextDocument.MarkdownFeature.MarkdownDialectGitHub) + local_time_asctimes + text + '\n')
        if len(self.outputText.toHtml()) > 1024 * 1024 * 10:
            self.outputText.clear()
        scrollbar: QScrollBar = self.outputText.verticalScrollBar()
        if scrollbar:
            scrollbar.setSliderPosition(scrollbar.maximum())


if __name__ == '__main__':
    use_modern_ui = False

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QtModernRedux.QApplication(sys.argv) if use_modern_ui else QApplication()
    app.setStyleSheet(open('resources/styleSheet.qss', encoding='utf-8').read())
    hbWin = HealBoneWindow()
    # 信号槽
    logSignal = LogSignal()
    logSignal.signal.connect(lambda log: hbWin.logViewAppend(log))
    logSignal.signal.emit("HealBone GaitStudio 初始化完成")

    if use_modern_ui:
        _hbWin = QtModernRedux.wrap(hbWin, titlebar_color=QColor('#303030'))
        _hbWin.show()
    else:
        hbWin.show()
    sys.exit(app.exec_())
