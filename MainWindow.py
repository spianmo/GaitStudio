# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from widgets.QVtkWidget import QVTKWidget
from widgets.QMaximumDockWidget import QMaximumDockWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1328, 803)
        MainWindow.setToolButtonStyle(Qt.ToolButtonIconOnly)
        MainWindow.setDocumentMode(False)
        MainWindow.setTabShape(QTabWidget.Rounded)
        MainWindow.setDockOptions(QMainWindow.AllowTabbedDocks|QMainWindow.AnimatedDocks)
        self.centralWidget = QWidget(MainWindow)
        self.centralWidget.setObjectName(u"centralWidget")
        self.horizontalLayout = QHBoxLayout(self.centralWidget)
        self.horizontalLayout.setSpacing(4)
        self.horizontalLayout.setContentsMargins(4, 4, 4, 4)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.tabWidget = QTabWidget(self.centralWidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.KinectIRFOV = QWidget()
        self.KinectIRFOV.setObjectName(u"KinectIRFOV")
        self.cameraIrFovView = QGraphicsView(self.KinectIRFOV)
        self.cameraIrFovView.setObjectName(u"cameraIrFovView")
        self.cameraIrFovView.setGeometry(QRect(0, 0, 741, 515))
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cameraIrFovView.sizePolicy().hasHeightForWidth())
        self.cameraIrFovView.setSizePolicy(sizePolicy)
        self.logoFrame = QFrame(self.KinectIRFOV)
        self.logoFrame.setObjectName(u"logoFrame")
        self.logoFrame.setGeometry(QRect(130, 160, 481, 191))
        self.logoFrame.setFrameShape(QFrame.Box)
        self.logoFrame.setFrameShadow(QFrame.Raised)
        self.tabWidget.addTab(self.KinectIRFOV, "")
        self.KinectFOV = QWidget()
        self.KinectFOV.setObjectName(u"KinectFOV")
        self.cameraFovView = QGraphicsView(self.KinectFOV)
        self.cameraFovView.setObjectName(u"cameraFovView")
        self.cameraFovView.setGeometry(QRect(0, 0, 734, 515))
        sizePolicy.setHeightForWidth(self.cameraFovView.sizePolicy().hasHeightForWidth())
        self.cameraFovView.setSizePolicy(sizePolicy)
        self.logoFrame_2 = QFrame(self.KinectFOV)
        self.logoFrame_2.setObjectName(u"logoFrame_2")
        self.logoFrame_2.setGeometry(QRect(130, 160, 481, 191))
        self.logoFrame_2.setFrameShape(QFrame.Box)
        self.logoFrame_2.setFrameShadow(QFrame.Raised)
        self.tabWidget.addTab(self.KinectFOV, "")
        self.KinectIR = QWidget()
        self.KinectIR.setObjectName(u"KinectIR")
        self.cameraIrView = QGraphicsView(self.KinectIR)
        self.cameraIrView.setObjectName(u"cameraIrView")
        self.cameraIrView.setGeometry(QRect(0, 0, 734, 515))
        sizePolicy.setHeightForWidth(self.cameraIrView.sizePolicy().hasHeightForWidth())
        self.cameraIrView.setSizePolicy(sizePolicy)
        self.logoFrame_3 = QFrame(self.KinectIR)
        self.logoFrame_3.setObjectName(u"logoFrame_3")
        self.logoFrame_3.setGeometry(QRect(130, 160, 481, 191))
        self.logoFrame_3.setFrameShape(QFrame.Box)
        self.logoFrame_3.setFrameShadow(QFrame.Raised)
        self.tabWidget.addTab(self.KinectIR, "")

        self.horizontalLayout.addWidget(self.tabWidget)

        MainWindow.setCentralWidget(self.centralWidget)
        self.statusBar = QStatusBar(MainWindow)
        self.statusBar.setObjectName(u"statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.resultsDock = QDockWidget(MainWindow)
        self.resultsDock.setObjectName(u"resultsDock")
        self.resultsDock.setEnabled(True)
        self.resultsDock.setMinimumSize(QSize(258, 143))
        self.resultsDock.setMaximumSize(QSize(640, 524287))
        font = QFont()
        font.setFamily(u"Arial")
        self.resultsDock.setFont(font)
        self.resultsDock.setFeatures(QDockWidget.DockWidgetFloatable|QDockWidget.DockWidgetMovable)
        self.resultsDock.setAllowedAreas(Qt.LeftDockWidgetArea|Qt.RightDockWidgetArea)
        self.resultsDockContents = QWidget()
        self.resultsDockContents.setObjectName(u"resultsDockContents")
        self.verticalLayout_5 = QVBoxLayout(self.resultsDockContents)
        self.verticalLayout_5.setSpacing(4)
        self.verticalLayout_5.setContentsMargins(4, 4, 4, 4)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(1, 0, 1, 1)
        self.groupBox = QGroupBox(self.resultsDockContents)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setEnabled(True)
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy1)
        self.groupBox.setMinimumSize(QSize(256, 0))
        self.btnStart = QPushButton(self.groupBox)
        self.btnStart.setObjectName(u"btnStart")
        self.btnStart.setGeometry(QRect(136, 30, 111, 32))
        self.sbTime = QSpinBox(self.groupBox)
        self.sbTime.setObjectName(u"sbTime")
        self.sbTime.setGeometry(QRect(15, 30, 111, 31))
        self.sbTime.setValue(3)
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(15, 12, 81, 16))
        font1 = QFont()
        font1.setFamily(u"Arial")
        font1.setPointSize(8)
        self.label.setFont(font1)
        self.cbFPS = QComboBox(self.groupBox)
        self.cbFPS.addItem("")
        self.cbFPS.addItem("")
        self.cbFPS.addItem("")
        self.cbFPS.setObjectName(u"cbFPS")
        self.cbFPS.setGeometry(QRect(15, 85, 111, 31))
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(15, 67, 81, 16))
        self.label_2.setFont(font1)
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(136, 67, 101, 16))
        self.label_3.setFont(font1)
        self.cbDepthMode = QComboBox(self.groupBox)
        self.cbDepthMode.addItem("")
        self.cbDepthMode.addItem("")
        self.cbDepthMode.addItem("")
        self.cbDepthMode.addItem("")
        self.cbDepthMode.setObjectName(u"cbDepthMode")
        self.cbDepthMode.setGeometry(QRect(136, 85, 111, 31))
        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(15, 120, 101, 16))
        self.label_4.setFont(font1)
        self.cbColorResolution = QComboBox(self.groupBox)
        self.cbColorResolution.addItem("")
        self.cbColorResolution.addItem("")
        self.cbColorResolution.addItem("")
        self.cbColorResolution.addItem("")
        self.cbColorResolution.addItem("")
        self.cbColorResolution.addItem("")
        self.cbColorResolution.setObjectName(u"cbColorResolution")
        self.cbColorResolution.setEnabled(True)
        self.cbColorResolution.setGeometry(QRect(15, 139, 111, 31))
        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(136, 119, 111, 16))
        self.label_5.setFont(font1)
        self.cbModelComplexity = QComboBox(self.groupBox)
        self.cbModelComplexity.addItem("")
        self.cbModelComplexity.addItem("")
        self.cbModelComplexity.addItem("")
        self.cbModelComplexity.setObjectName(u"cbModelComplexity")
        self.cbModelComplexity.setGeometry(QRect(136, 137, 111, 31))
        self.hsMinDetectionConfidence = QSlider(self.groupBox)
        self.hsMinDetectionConfidence.setObjectName(u"hsMinDetectionConfidence")
        self.hsMinDetectionConfidence.setGeometry(QRect(15, 247, 231, 22))
        self.hsMinDetectionConfidence.setMaximum(100)
        self.hsMinDetectionConfidence.setPageStep(1)
        self.hsMinDetectionConfidence.setValue(50)
        self.hsMinDetectionConfidence.setSliderPosition(50)
        self.hsMinDetectionConfidence.setOrientation(Qt.Horizontal)
        self.tvMinDetectionConfidence = QLabel(self.groupBox)
        self.tvMinDetectionConfidence.setObjectName(u"tvMinDetectionConfidence")
        self.tvMinDetectionConfidence.setGeometry(QRect(15, 226, 221, 16))
        self.tvMinDetectionConfidence.setFont(font1)
        self.tvMinTrackingConfidence = QLabel(self.groupBox)
        self.tvMinTrackingConfidence.setObjectName(u"tvMinTrackingConfidence")
        self.tvMinTrackingConfidence.setGeometry(QRect(15, 176, 211, 16))
        self.tvMinTrackingConfidence.setFont(font1)
        self.hsMinTrackingConfidence = QSlider(self.groupBox)
        self.hsMinTrackingConfidence.setObjectName(u"hsMinTrackingConfidence")
        self.hsMinTrackingConfidence.setGeometry(QRect(15, 197, 231, 22))
        self.hsMinTrackingConfidence.setMaximum(100)
        self.hsMinTrackingConfidence.setPageStep(1)
        self.hsMinTrackingConfidence.setValue(50)
        self.hsMinTrackingConfidence.setSliderPosition(50)
        self.hsMinTrackingConfidence.setOrientation(Qt.Horizontal)
        self.cbSmoothLandmarks = QCheckBox(self.groupBox)
        self.cbSmoothLandmarks.setObjectName(u"cbSmoothLandmarks")
        self.cbSmoothLandmarks.setGeometry(QRect(15, 277, 171, 21))
        self.cbSmoothLandmarks.setFont(font1)
        self.cbSmoothLandmarks.setChecked(True)
        self.cbSmoothLandmarks.setTristate(False)
        self.tvAnglesRangeFrames = QLabel(self.groupBox)
        self.tvAnglesRangeFrames.setObjectName(u"tvAnglesRangeFrames")
        self.tvAnglesRangeFrames.setGeometry(QRect(15, 314, 201, 16))
        self.tvAnglesRangeFrames.setFont(font1)
        self.hsAnglesViewerRange = QSlider(self.groupBox)
        self.hsAnglesViewerRange.setObjectName(u"hsAnglesViewerRange")
        self.hsAnglesViewerRange.setGeometry(QRect(15, 337, 231, 22))
        self.hsAnglesViewerRange.setMinimum(10)
        self.hsAnglesViewerRange.setMaximum(120)
        self.hsAnglesViewerRange.setPageStep(1)
        self.hsAnglesViewerRange.setValue(60)
        self.hsAnglesViewerRange.setSliderPosition(60)
        self.hsAnglesViewerRange.setOrientation(Qt.Horizontal)
        self.tvAnglesItemVisibleNum = QLabel(self.groupBox)
        self.tvAnglesItemVisibleNum.setObjectName(u"tvAnglesItemVisibleNum")
        self.tvAnglesItemVisibleNum.setGeometry(QRect(15, 364, 201, 16))
        self.tvAnglesItemVisibleNum.setFont(font1)
        self.hsAnglesViewerItems = QSlider(self.groupBox)
        self.hsAnglesViewerItems.setObjectName(u"hsAnglesViewerItems")
        self.hsAnglesViewerItems.setGeometry(QRect(15, 387, 231, 22))
        self.hsAnglesViewerItems.setMinimum(10)
        self.hsAnglesViewerItems.setMaximum(80)
        self.hsAnglesViewerItems.setPageStep(1)
        self.hsAnglesViewerItems.setValue(30)
        self.hsAnglesViewerItems.setSliderPosition(30)
        self.hsAnglesViewerItems.setOrientation(Qt.Horizontal)
        self.cbPatientBtn = QCheckBox(self.groupBox)
        self.cbPatientBtn.setObjectName(u"cbPatientBtn")
        self.cbPatientBtn.setGeometry(QRect(140, 422, 91, 21))
        self.cbPatientBtn.setFont(font1)
        self.cbPatientBtn.setChecked(False)
        self.cbPatientBtn.setTristate(False)
        self.cbPlotAlignBtn = QCheckBox(self.groupBox)
        self.cbPlotAlignBtn.setObjectName(u"cbPlotAlignBtn")
        self.cbPlotAlignBtn.setGeometry(QRect(15, 422, 111, 21))
        self.cbPlotAlignBtn.setFont(font1)
        self.cbPlotAlignBtn.setChecked(True)
        self.cbPlotAlignBtn.setTristate(False)

        self.verticalLayout_5.addWidget(self.groupBox)

        self.resultsDock.setWidget(self.resultsDockContents)
        MainWindow.addDockWidget(Qt.LeftDockWidgetArea, self.resultsDock)
        self.messagesDock = QDockWidget(MainWindow)
        self.messagesDock.setObjectName(u"messagesDock")
        self.messagesDock.setMinimumSize(QSize(91, 101))
        self.messagesDock.setFeatures(QDockWidget.DockWidgetFloatable|QDockWidget.DockWidgetMovable)
        self.messagesDockContents = QWidget()
        self.messagesDockContents.setObjectName(u"messagesDockContents")
        self.verticalLayout_3 = QVBoxLayout(self.messagesDockContents)
        self.verticalLayout_3.setSpacing(4)
        self.verticalLayout_3.setContentsMargins(4, 4, 4, 4)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(1, 0, 1, 1)
        self.outputText = QTextEdit(self.messagesDockContents)
        self.outputText.setObjectName(u"outputText")
        self.outputText.setFont(font)
        self.outputText.setFrameShape(QFrame.StyledPanel)
        self.outputText.setFrameShadow(QFrame.Plain)
        self.outputText.setReadOnly(True)

        self.verticalLayout_3.addWidget(self.outputText)

        self.messagesDock.setWidget(self.messagesDockContents)
        MainWindow.addDockWidget(Qt.BottomDockWidgetArea, self.messagesDock)
        self.viewerDock = QDockWidget(MainWindow)
        self.viewerDock.setObjectName(u"viewerDock")
        self.viewerDock.setFloating(False)
        self.viewerDock.setFeatures(QDockWidget.DockWidgetFloatable|QDockWidget.DockWidgetMovable)
        self.viewerDockContents = QWidget()
        self.viewerDockContents.setObjectName(u"viewerDockContents")
        self.verticalLayout = QVBoxLayout(self.viewerDockContents)
        self.verticalLayout.setSpacing(4)
        self.verticalLayout.setContentsMargins(4, 4, 4, 4)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.vtkContext = QVTKWidget(self.viewerDockContents)
        self.vtkContext.setObjectName(u"vtkContext")
        sizePolicy.setHeightForWidth(self.vtkContext.sizePolicy().hasHeightForWidth())
        self.vtkContext.setSizePolicy(sizePolicy)
        self.vtkContext.setMinimumSize(QSize(322, 0))

        self.verticalLayout.addWidget(self.vtkContext)

        self.viewerDock.setWidget(self.viewerDockContents)
        MainWindow.addDockWidget(Qt.RightDockWidgetArea, self.viewerDock)
        self.angleViewerDock = QMaximumDockWidget(MainWindow)
        self.angleViewerDock.setObjectName(u"angleViewerDock")
        self.angleViewerDock.setFloating(False)
        self.angleViewerDock.setFeatures(QDockWidget.DockWidgetFloatable|QDockWidget.DockWidgetMovable)
        self.angleViewerDockContents = QWidget()
        self.angleViewerDockContents.setObjectName(u"angleViewerDockContents")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.angleViewerDockContents.sizePolicy().hasHeightForWidth())
        self.angleViewerDockContents.setSizePolicy(sizePolicy2)
        self.angleViewerDockContents.setMinimumSize(QSize(0, 0))
        self.angleViewerDockContents.setMaximumSize(QSize(16777215, 16777215))
        self.angleViewerDockContentsVerticalLayout = QVBoxLayout(self.angleViewerDockContents)
        self.angleViewerDockContentsVerticalLayout.setSpacing(4)
        self.angleViewerDockContentsVerticalLayout.setContentsMargins(4, 4, 4, 4)
        self.angleViewerDockContentsVerticalLayout.setObjectName(u"angleViewerDockContentsVerticalLayout")
        self.angleViewerDockContentsVerticalLayout.setContentsMargins(0, 0, 0, 0)
        self.angleViewerDockScrollArea = QScrollArea(self.angleViewerDockContents)
        self.angleViewerDockScrollArea.setObjectName(u"angleViewerDockScrollArea")
        self.angleViewerDockScrollArea.setWidgetResizable(True)
        self.angleViewerDockScrollAreaContents = QWidget()
        self.angleViewerDockScrollAreaContents.setObjectName(u"angleViewerDockScrollAreaContents")
        self.angleViewerDockScrollAreaContents.setGeometry(QRect(0, 0, 320, 356))
        self.angleViewerDockScrollAreaContentsLayout = QVBoxLayout(self.angleViewerDockScrollAreaContents)
        self.angleViewerDockScrollAreaContentsLayout.setSpacing(4)
        self.angleViewerDockScrollAreaContentsLayout.setContentsMargins(4, 4, 4, 4)
        self.angleViewerDockScrollAreaContentsLayout.setObjectName(u"angleViewerDockScrollAreaContentsLayout")
        self.angleViewerDockScrollAreaContentsLayout.setContentsMargins(0, 0, 0, 0)
        self.angleViewerDockScrollArea.setWidget(self.angleViewerDockScrollAreaContents)

        self.angleViewerDockContentsVerticalLayout.addWidget(self.angleViewerDockScrollArea)

        self.angleViewerDock.setWidget(self.angleViewerDockContents)
        MainWindow.addDockWidget(Qt.RightDockWidgetArea, self.angleViewerDock)
        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 1328, 23))
        MainWindow.setMenuBar(self.menuBar)
        self.anglesDataFrameDock = QDockWidget(MainWindow)
        self.anglesDataFrameDock.setObjectName(u"anglesDataFrameDock")
        self.anglesDataFrameDock.setMinimumSize(QSize(100, 38))
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        self.dockWidgetContents.setMinimumSize(QSize(100, 0))
        self.anglesDataFrameDock.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(Qt.BottomDockWidgetArea, self.anglesDataFrameDock)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)
        self.cbFPS.setCurrentIndex(2)
        self.cbColorResolution.setCurrentIndex(1)
        self.cbModelComplexity.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"HealBone GaitStudio", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.KinectIRFOV), QCoreApplication.translate("MainWindow", u"IR FOV", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.KinectFOV), QCoreApplication.translate("MainWindow", u"FOV", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.KinectIR), QCoreApplication.translate("MainWindow", u"IR", None))
        self.resultsDock.setWindowTitle(QCoreApplication.translate("MainWindow", u"Gait Control Panel", None))
        self.groupBox.setTitle("")
        self.btnStart.setText(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb\u6b65\u6001\u68c0\u6d4b", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u68c0\u6d4b\u65f6\u957f/sec", None))
        self.cbFPS.setItemText(0, QCoreApplication.translate("MainWindow", u"FPS_5", None))
        self.cbFPS.setItemText(1, QCoreApplication.translate("MainWindow", u"FPS_15", None))
        self.cbFPS.setItemText(2, QCoreApplication.translate("MainWindow", u"FPS_30", None))

        self.label_2.setText(QCoreApplication.translate("MainWindow", u"FPS", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"DepthMode", None))
        self.cbDepthMode.setItemText(0, QCoreApplication.translate("MainWindow", u"NFOV_2X2BD", None))
        self.cbDepthMode.setItemText(1, QCoreApplication.translate("MainWindow", u"NFOV_UNBD", None))
        self.cbDepthMode.setItemText(2, QCoreApplication.translate("MainWindow", u"WFOV_2X2BD", None))
        self.cbDepthMode.setItemText(3, QCoreApplication.translate("MainWindow", u"WFOV_UNBD", None))

        self.label_4.setText(QCoreApplication.translate("MainWindow", u"ColorResolution", None))
        self.cbColorResolution.setItemText(0, QCoreApplication.translate("MainWindow", u"RES_720P", None))
        self.cbColorResolution.setItemText(1, QCoreApplication.translate("MainWindow", u"RES_1080P", None))
        self.cbColorResolution.setItemText(2, QCoreApplication.translate("MainWindow", u"RES_1440P", None))
        self.cbColorResolution.setItemText(3, QCoreApplication.translate("MainWindow", u"RES_1536P", None))
        self.cbColorResolution.setItemText(4, QCoreApplication.translate("MainWindow", u"RES_2160P", None))
        self.cbColorResolution.setItemText(5, QCoreApplication.translate("MainWindow", u"RES_3072P", None))

        self.label_5.setText(QCoreApplication.translate("MainWindow", u"ModelComplexity", None))
        self.cbModelComplexity.setItemText(0, QCoreApplication.translate("MainWindow", u"LOW", None))
        self.cbModelComplexity.setItemText(1, QCoreApplication.translate("MainWindow", u"MIDDLE", None))
        self.cbModelComplexity.setItemText(2, QCoreApplication.translate("MainWindow", u"HIGH", None))

        self.tvMinDetectionConfidence.setText(QCoreApplication.translate("MainWindow", u"Min Detection Confidence (0.5)", None))
        self.tvMinTrackingConfidence.setText(QCoreApplication.translate("MainWindow", u"Min Tracking Confidence (0.5)", None))
        self.cbSmoothLandmarks.setText(QCoreApplication.translate("MainWindow", u"Smooth Landmarks", None))
        self.tvAnglesRangeFrames.setText(QCoreApplication.translate("MainWindow", u"Angles-Viewer Range (60 Frames)", None))
        self.tvAnglesItemVisibleNum.setText(QCoreApplication.translate("MainWindow", u"Angle-Items Visible Num (3 Items)", None))
        self.cbPatientBtn.setText(QCoreApplication.translate("MainWindow", u"Patient Mode", None))
        self.cbPlotAlignBtn.setText(QCoreApplication.translate("MainWindow", u"Plot Align Button", None))
        self.messagesDock.setWindowTitle(QCoreApplication.translate("MainWindow", u"Analysis Messages", None))
        self.viewerDock.setWindowTitle(QCoreApplication.translate("MainWindow", u"RealTime Human Skeleton Viewer", None))
        self.angleViewerDock.setWindowTitle(QCoreApplication.translate("MainWindow", u"RealTime Angles Viewer", None))
        self.anglesDataFrameDock.setWindowTitle(QCoreApplication.translate("MainWindow", u"Angles DataFrame", None))
    # retranslateUi

