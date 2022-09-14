import os
import sys

from PySide2 import QtWebEngineWidgets
from PySide2.QtCore import *
from PySide2.QtWidgets import *

def main():
    app = QApplication()
    view = QtWebEngineWidgets.QWebEngineView()
    settings = view.settings()
    settings.setAttribute(QtWebEngineWidgets.QWebEngineSettings.PluginsEnabled, True)
    settings.setAttribute(QtWebEngineWidgets.QWebEngineSettings.PdfViewerEnabled, True)
    url = QUrl.fromLocalFile(f"{os.path.abspath('.')}{'/calib.io_checker_3000x2000_5x8_319.pdf'}")
    view.load(url)
    view.resize(640, 480)
    view.show()
    view.page().printToPdf("test.pdf")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
