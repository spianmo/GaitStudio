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
