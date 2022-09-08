import sys
import PySide2
from PySide2.QtCore import QEvent, Qt
from PySide2.QtWidgets import QApplication, QWidget, QDockWidget


class QMaximumDockWidget(QDockWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

    def event(self, event: QEvent) -> bool:
        if QEvent.ZOrderChange == event.type():
            if self.isFloating():
                w: QWidget = QWidget()
                self.setMaximumSize(w.maximumSize())
                self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
                self.show()
        return super().event(event)
