import sys
import PySide2
from PySide2.QtWidgets import QApplication, QWidget


class QVTKWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
