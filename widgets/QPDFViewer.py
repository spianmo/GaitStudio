import os
import sys
import tempfile


from PySide2 import QtWebEngineWidgets
from PySide2.QtCore import QUrl, Qt
from PySide2.QtGui import QPainter
from PySide2.QtPrintSupport import QPrinter, QPrintDialog
from PySide2.QtWidgets import *
from PIL.ImageQt import ImageQt
from pdf2image import convert_from_path
from qtmodernredux import QtModernRedux


class PDFViewer(QDialog):
    def __init__(self, title="HealBone Report", pdf=""):
        super().__init__()
        self.title = title
        self.pdf = f"{os.path.abspath('.')}/{pdf}"
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle(self.title)
        self.setFixedSize(640, 900)
        self.layout = QVBoxLayout()
        self.layout.setMargin(0)

        self.menuBar = QMenuBar()

        self.localfile_action = QAction(self.menuBar)
        self.localfile_action.setText("打印报告")
        self.localfile_action.setCheckable(False)
        self.localfile_action.triggered.connect(self.printPdf)
        self.menuBar.addAction(self.localfile_action)

        self.webview = QtWebEngineWidgets.QWebEngineView()
        settings = self.webview.settings()
        settings.setAttribute(QtWebEngineWidgets.QWebEngineSettings.PluginsEnabled, True)
        settings.setAttribute(QtWebEngineWidgets.QWebEngineSettings.PdfViewerEnabled, True)
        settings.setAttribute(QtWebEngineWidgets.QWebEngineSettings.WebAttribute.ShowScrollBars, False)
        url = QUrl.fromLocalFile(self.pdf)
        self.webview.load(url)
        self.webview.resize(640, 480)
        self.webview.show()
        self.layout.setMenuBar(self.menuBar)
        self.layout.addWidget(self.webview)
        self.setLayout(self.layout)

    def printPdf(self):
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)
        if dialog.exec_() == QPrintDialog.Accepted:
            with tempfile.TemporaryDirectory() as path:
                images = convert_from_path(self.pdf, dpi=300, output_folder=path, poppler_path=f"{os.path.abspath('.')}/../resources/poppler-22"
                                                                                               f".04.0/Library/bin")
                painter = QPainter()
                painter.begin(printer)
                for i, image in enumerate(images):
                    if i > 0:
                        printer.newPage()
                    rect = painter.viewport()
                    qtImage = ImageQt(image)
                    qtImageScaled = qtImage.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    painter.drawImage(rect, qtImageScaled)
                painter.end()


if __name__ == '__main__':
    app = QtModernRedux.QApplication(sys.argv)
    app.setStyleSheet(open('../resources/styleSheet.qss', encoding='utf-8').read())
    data_name = "GaitReport-20220914151120"
    v = QtModernRedux.wrap(PDFViewer(title=data_name, pdf=f'../report_output/{data_name}.pdf'), transparent_window=False)
    if v.exec_():
        print("")
    sys.exit(app.exec_())
