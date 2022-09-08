import sys
import time
from typing import List

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import pyqtgraph as pg
from pyqtgraph import FillBetweenItem

import MainWindow
from GUISignal import LogSignal
from KinectCameraThread import KinectCaptureThread
import cv2 as cv

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
        anglesViewerRange = 60
        patientMode = False

    def __init__(self, *args):
        QMainWindow.__init__(self, *args)
        MainWindow.Ui_MainWindow.__init__(self)
        self.threadCapture: KinectCaptureThread = None
        self.viewModel = self.HealBoneViewModel()
        # self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
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
                self.anglePltLayouts[anglesCubeIndex].append(pg.PlotWidget(parent=self.angleViewerDockScrollAreaContents, background="#2F2F2F"))
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setTitle(anglesCube["title"] + " " + angleCube[1], **label_style)
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setAntialiasing(True)
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setXRange(0, 100)
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setYRange(anglesCube["ylim"][0], anglesCube["ylim"][1])
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].showGrid(x=True, y=True)

                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setMinimumHeight(180)

                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setLabel('left', angleCube[3], **label_style)
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setLabel('bottom', angleCube[2], **label_style)
                self.angleViewerDockScrollAreaContentsLayout.addWidget(self.anglePltLayouts[anglesCubeIndex][angleCubeIndex])
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
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setXLink(self.anglePltLayouts[anglesCubeIndex][angleCubeIndex - 1])

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
        self.patientWin.setFloating(True)
        self.patientWin.setHidden(True)
        self.patientWin.setMinimumSize(QSize(741, 515))
        self.showStatusMessage("Info: Healbone GaitStudio组件加载完毕")

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
        self.tvAnglesRangeFrames.setText("Angles-Viewer Range (" + str(self.hsAnglesViewerRange.value()) + " Frames)")
        self.viewModel.anglesViewerRange = int(self.hsAnglesViewerRange.value())

    def changeMinTrackingConfidence(self):
        self.tvMinTrackingConfidence.setText("Min Tracking Confidence (" + str(round(self.hsMinTrackingConfidence.value() / 100, 1)) + ")")

    def changeMinDetectionConfidence(self):
        self.tvMinDetectionConfidence.setText("Min Detection Confidence (" + str(round(self.hsMinDetectionConfidence.value() / 100, 1)) + ")")

    def changeHsAnglesViewerItems(self):
        for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
            for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                self.tvAnglesItemVisibleNum.setText("Angle-Items Visible Num (" + str(int(self.hsAnglesViewerItems.value() / 10)) + " Items)")
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setMinimumHeight(
                    int(self.angleViewerDock.size().height() / int(self.hsAnglesViewerItems.value() / 10) - 20))

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Resize and watched == self.tabWidget:
            self.resizeCameraView()
        if event.type() == QEvent.WindowStateChange and watched == self.angleViewerDock:
            if self.angleViewerDock.isFloating() and event.type() == QEvent.Resize and watched == self.angleViewerDock:
                for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
                    for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                        self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setMinimumHeight(int(event.size().height() / 6 - 20))
            if not self.angleViewerDock.isFloating() and event.type() == QEvent.Resize and watched == self.angleViewerDock:
                for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
                    for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                        self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].setMinimumHeight(int(event.size().height() / 3 - 20))
        return super().eventFilter(watched, event)

    def resizeCameraView(self, tabWidth=-1, tabHeight=-1):
        tabWidgetSize: QSize = self.tabWidget.geometry().size()
        self.cameraIrFovView.setGeometry(self.cameraIrFovView.x(), self.cameraIrFovView.y(), tabWidgetSize.width() if tabWidth == -1 else tabWidth,
                                         (tabWidgetSize.height() - 24) if tabHeight == -1 else tabHeight)
        self.cameraFovView.setGeometry(self.cameraFovView.x(), self.cameraFovView.y(), tabWidgetSize.width() if tabWidth == -1 else tabWidth,
                                       (tabWidgetSize.height() - 24) if tabHeight == -1 else tabHeight)
        self.cameraIrView.setGeometry(self.cameraIrView.x(), self.cameraIrView.y(), tabWidgetSize.width() if tabWidth == -1 else tabWidth,
                                      (tabWidgetSize.height() - 24) if tabHeight == -1 else tabHeight)

    def plotFrameAngles(self, angles: dict):
        for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
            for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                self.anglePltDataList[anglesCubeIndex][angleCubeIndex][0].append(angles["time_index"])
                self.anglePltDataList[anglesCubeIndex][angleCubeIndex][1].append(angles[angleCube[1]])
                if len(self.anglePltDataList[anglesCubeIndex][angleCubeIndex][0]) > self.viewModel.anglesViewerRange:
                    self.anglePltDataList[anglesCubeIndex][angleCubeIndex][0].pop(0)
                if len(self.anglePltDataList[anglesCubeIndex][angleCubeIndex][1]) > self.viewModel.anglesViewerRange:
                    self.anglePltDataList[anglesCubeIndex][angleCubeIndex][1].pop(0)
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].getPlotItem().plot(self.anglePltDataList[anglesCubeIndex][angleCubeIndex][0],
                                                                                         self.anglePltDataList[anglesCubeIndex][angleCubeIndex][1],
                                                                                         pen=({'color': "r", "width": 1.5}), clear=True)

    def clearAnglesViewer(self):
        for anglesCubeIndex, anglesCube in enumerate(self.viewModel.anglesCheckCube):
            for angleCubeIndex, angleCube in enumerate(anglesCube["axis"]):
                self.anglePltDataList[anglesCubeIndex][angleCubeIndex] = [[], []]
                self.anglePltLayouts[anglesCubeIndex][angleCubeIndex].getPlotItem().plot(self.anglePltDataList[anglesCubeIndex][angleCubeIndex][0],
                                                                                         self.anglePltDataList[anglesCubeIndex][angleCubeIndex][1],
                                                                                         pen=({'color': "r", "width": 1.5}), clear=True)

    def displayCVFrames(self, frames):
        self.displayCVFrame(self.cameraIrFovView, frames[2])
        self.displayCVFrame(self.cameraFovView, frames[1])
        self.displayCVFrame(self.cameraIrView, frames[0])
        self.displayCVFrame(self.cameraPatientView, frames[0])

    def displayCVFrame(self, cameraView, frame):
        """
        将cv的frame显示到label上
        """
        shrink = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        QtImg = QImage(shrink.data,
                       shrink.shape[1],
                       shrink.shape[0],
                       shrink.shape[1] * 3,
                       QImage.Format_RGB888)
        jpg_out = QPixmap(QtImg).scaled(cameraView.width(), cameraView.height(), Qt.KeepAspectRatioByExpanding)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(QGraphicsPixmapItem(jpg_out))
        cameraView.setScene(scene)  # 将场景添加至视图

    def stopDetect(self):
        self.viewModel.detectStatus = False
        self.threadCapture.stopCapture()
        self.btnStart.setText("开始检测")

    def startDetect(self, k4aConfig, mpConfig, captureConfig):
        self.viewModel.detectStatus = True
        self.btnStart.setText("停止检测")
        self.threadCapture = KinectCaptureThread(k4aConfig=k4aConfig, mpConfig=mpConfig, captureConfig=captureConfig)
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
        self.threadCapture.start()

    def enableDetectForm(self, enable=True):
        self.sbTime.setEnabled(enable)
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
        self.logViewAppend("Pose Detect完成, 结果分析中...")

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
                "min_detection_confidence": round(self.hsMinDetectionConfidence.sliderPosition() / self.hsMinDetectionConfidence.maximum(), 1),
                "min_tracking_confidence": round(self.hsMinTrackingConfidence.sliderPosition() / self.hsMinTrackingConfidence.maximum(), 1),
                "model_complexity": self.cbModelComplexity.currentIndex(),
                "smooth_landmarks": self.cbSmoothLandmarks.isChecked()
            }
            captureConfig = {
                "fps": int(self.cbFPS.currentText().split("_")[1]),
                "detectionTime": int(self.sbTime.text())
            }
            self.clearAnglesViewer()
            self.logoFrame.hide()
            self.logoFrame_2.hide()
            self.logoFrame_3.hide()
            self.startDetect(k4aConfig, mpConfig, captureConfig)
            if self.viewModel.patientMode:
                self.patientWin.show()
            self.enableDetectForm(enable=False)

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
    app = QApplication()
    app.setStyleSheet(open('resources/styleSheet.qss', encoding='utf-8').read())
    hbWin = HealBoneWindow()
    # 信号槽
    logSignal = LogSignal()
    logSignal.signal.connect(lambda log: hbWin.logViewAppend(log))
    logSignal.signal.emit("HealBone GaitStudio 初始化完成")

    hbWin.show()
    sys.exit(app.exec_())
