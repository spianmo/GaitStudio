import time

from PySide2 import QtCore
from PySide2.QtWidgets import *
from PySide2.QtGui import *
import pandas as pd
from qtmodernredux import QtModernRedux
from tablexplore import core, plotting, interpreter
from tablexplore.util import getSampleData


class TestApp(QMainWindow):
    def __init__(self, project_file=None, csv_file=None):
        QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Example")
        self.setGeometry(QtCore.QRect(200, 200, 800, 600))
        self.main = QWidget()
        self.setCentralWidget(self.main)
        layout = QVBoxLayout(self.main)
        df = getSampleData()
        t = core.DataFrameWidget(self.main, dataframe=df)
        layout.addWidget(t)
        # show a Python interpreter
        # t.showInterpreter()
        # t.showFullScreen()
        return


if __name__ == '__main__':
    import sys

    app = QtModernRedux.QApplication(sys.argv)
    aw = QtModernRedux.wrap(TestApp())
    aw.show()
    app.exec_()


