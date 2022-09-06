from PySide2.QtCore import QObject, Signal


class LogSignal(QObject):
    signal = Signal(str)
