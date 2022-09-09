from typing import List

from PySide2.QtCore import QObject, Signal
from PySide2.QtGui import QPixmap


class LogSignal(QObject):
    signal = Signal(str)


class VideoFramesSignal(QObject):
    signal = Signal(list)


class KeyPointsSignal(QObject):
    signal = Signal(list)


class AngleDictSignal(QObject):
    signal = Signal(dict)


class DetectInterruptSignal(QObject):
    signal = Signal(str)


class DetectFinishSignal(QObject):
    signal = Signal(str)


class DetectExitSignal(QObject):
    signal = Signal(str)

class KinectErrorSignal(QObject):
    signal = Signal(str)