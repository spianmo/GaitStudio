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


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1294, 623)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(0, 0, 1291, 621))
        self.rootContainer = QHBoxLayout(self.horizontalLayoutWidget)
        self.rootContainer.setObjectName(u"rootContainer")
        self.rootContainer.setContentsMargins(0, 0, 0, 0)
        self.cameraView = QGraphicsView(self.horizontalLayoutWidget)
        self.cameraView.setObjectName(u"cameraView")
        self.cameraView.setMinimumSize(QSize(920, 0))
        self.cameraView.setMaximumSize(QSize(557, 16777215))

        self.rootContainer.addWidget(self.cameraView)

        self.rightPanel = QVBoxLayout()
        self.rightPanel.setObjectName(u"rightPanel")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.groupBox = QGroupBox(self.horizontalLayoutWidget)
        self.groupBox.setObjectName(u"groupBox")
        self.sbTime = QSpinBox(self.groupBox)
        self.sbTime.setObjectName(u"sbTime")
        self.sbTime.setGeometry(QRect(90, 40, 111, 31))
        self.sbTime.setValue(4)
        self.btnStart = QPushButton(self.groupBox)
        self.btnStart.setObjectName(u"btnStart")
        self.btnStart.setGeometry(QRect(230, 40, 121, 31))
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 50, 61, 16))
        font = QFont()
        font.setPointSize(11)
        self.label.setFont(font)

        self.horizontalLayout_2.addWidget(self.groupBox)

        self.rightPanel.addLayout(self.horizontalLayout_2)

        self.rootContainer.addLayout(self.rightPanel)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"HealBone-GaitAnalysis", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Control Panel", None))
        self.btnStart.setText(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb\u6b65\u6001\u68c0\u6d4b", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u68c0\u6d4b\u65f6\u957f", None))
    # retranslateUi
