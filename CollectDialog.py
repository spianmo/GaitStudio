# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'CollectDialog.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_CollectDialog(object):
    def setupUi(self, CollectDialog):
        if not CollectDialog.objectName():
            CollectDialog.setObjectName(u"CollectDialog")
        CollectDialog.resize(322, 165)
        self.formLayoutWidget = QWidget(CollectDialog)
        self.formLayoutWidget.setObjectName(u"formLayoutWidget")
        self.formLayoutWidget.setGeometry(QRect(0, 0, 321, 121))
        self.formLayout = QFormLayout(self.formLayoutWidget)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonBox = QDialogButtonBox(CollectDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setGeometry(QRect(0, 130, 311, 31))
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)

        self.retranslateUi(CollectDialog)
        self.buttonBox.rejected.connect(CollectDialog.reject)
        self.buttonBox.accepted.connect(CollectDialog.accept)

        QMetaObject.connectSlotsByName(CollectDialog)
    # setupUi

    def retranslateUi(self, CollectDialog):
        CollectDialog.setWindowTitle(QCoreApplication.translate("CollectDialog", u"\u586b\u5199\u672c\u6b21\u8bc4\u4f30\u6240\u9700\u57fa\u672c\u4fe1\u606f", None))
    # retranslateUi

