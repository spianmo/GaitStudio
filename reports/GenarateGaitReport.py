import time
from io import BytesIO
from typing import List, Tuple

from matplotlib import pyplot as plt
from pandas import DataFrame
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (SimpleDocTemplate, Paragraph, PageBreak, Image, Spacer, Table, TableStyle)
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import LETTER, inch
from reportlab.graphics.shapes import Line, LineShape, Drawing
from reportlab.lib.colors import Color
from svglib.svglib import svg2rlg
import seaborn as sns

resourcePath = 'reports/'


class PageHeaderFooter(canvas.Canvas):

    def __init__(self, *args, **kwargs):
        self.pages = []
        self.width, self.height = LETTER
        canvas.Canvas.__init__(self, *args, **kwargs)

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_count = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            if self._pageNumber > 1:
                self.drawPageHeaderFooter(page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def drawPageHeaderFooter(self, page_count):
        page = "Page %s of %s" % (self._pageNumber, page_count)
        x = 128
        self.saveState()
        self.setStrokeColorRGB(0, 0, 0)
        self.setLineWidth(0.5)
        self.drawImage(resourcePath + "static/healbone_banner.png", self.width - inch * 8 - 5, self.height - 50, width=100, height=20,
                       preserveAspectRatio=True,
                       mask='auto')
        self.drawImage(resourcePath + "static/gait.png", self.width - inch * 2, self.height - 50, width=100, height=30, preserveAspectRatio=True,
                       mask='auto')
        self.line(30, 740, LETTER[0] - 50, 740)
        self.line(66, 78, LETTER[0] - 66, 78)
        self.setFont('Times-Roman', 10)
        self.drawString(LETTER[0] - x, 65, page)
        self.restoreState()


colorGreen0 = Color((45.0 / 255), (166.0 / 255), (153.0 / 255), 1)
colorGreen1 = Color((182.0 / 255), (227.0 / 255), (166.0 / 255), 1)
colorGreen2 = Color((140.0 / 255), (222.0 / 255), (192.0 / 255), 1)
colorBlue0 = Color((54.0 / 255), (122.0 / 255), (179.0 / 255), 1)
colorBlack = Color((0.0 / 255), (0.0 / 255), (0.0 / 255), 1)
colorBlue1 = Color((122.0 / 255), (180.0 / 255), (225.0 / 255), 1)
colorGreenLine = Color((50.0 / 255), (140.0 / 255), (140.0 / 255), 1)


def ParagraphReportHeader(color=colorGreen0, fontSize=12, text=''):
    psHeaderText = ParagraphStyle('Hed0', fontName='msyh', fontSize=fontSize, alignment=TA_LEFT, borderWidth=3, textColor=color)
    return Paragraph(text, psHeaderText)


def ReportDivider(color=colorGreenLine):
    d = Drawing(500, 1)
    line = Line(-15, 0, 483, 0)
    line.strokeColor = color
    line.strokeWidth = 2
    d.add(line)
    return d


def HeightSpacer(heightPixels=10):
    return Spacer(1, heightPixels)


def ReportNoBorderTable(lineData: List[List], colWidths=None, rowHeights=None, tableStyle: List[Tuple] = []):
    table = Table(lineData, colWidths=colWidths, rowHeights=rowHeights)
    tStyle = TableStyle(tableStyle)
    table.setStyle(tStyle)
    return table


def generate_angle_plots(df: DataFrame) -> List[BytesIO]:
    metadatas = [
        {
            "title": "躯干髋关节角度",
            "axis": [
                ["Time_in_sec", "TorsoLHip_angle", "时间（秒）", "躯干 L 髋关节角度 (°)"],
                ["Time_in_sec", "TorsoRHip_angle", "时间（秒）", "躯干 R 髋关节角度 (°)"]
            ]
        },
        {
            "title": "髋关节角度",
            "axis": [
                ["Time_in_sec", "LHip_angle", "时间（秒）", "L 髋关节角度 (°)"],
                ["Time_in_sec", "RHip_angle", "时间（秒）", "R 髋关节角度 (°)"]
            ]
        },
        {
            "title": "膝关节角度",
            "axis": [
                ["Time_in_sec", "LKnee_angle", "时间（秒）", "L 膝关节角度 (°)"],
                ["Time_in_sec", "RKnee_angle", "时间（秒）", "R 膝关节角度 (°)"]
            ]
        },
        {
            "title": "踝关节角度",
            "axis": [
                ["Time_in_sec", "LAnkle_angle", "时间（秒）", "L 踝关节角度 (°)"],
                ["Time_in_sec", "RAnkle_angle", "时间（秒）", "R 踝关节角度 (°)"]
            ]
        }
    ]
    images = []
    rc = {'font.sans-serif': 'SimHei',
          'axes.unicode_minus': False}
    sns.set_style(style='darkgrid', rc=rc)
    for metadata in metadatas:
        fig, axes = plt.subplots(2, 1, figsize=(5.5, 7))

        fig.suptitle(metadata["title"])

        sns.lineplot(ax=axes[0], data=df, x=metadata["axis"][0][0], y=metadata["axis"][0][1]).set(xlabel=metadata["axis"][0][2],
                                                                                                  ylabel=metadata["axis"][0][3])

        sns.lineplot(ax=axes[1], data=df, x=metadata["axis"][1][0], y=metadata["axis"][1][1]).set(xlabel=metadata["axis"][1][2],
                                                                                                  ylabel=metadata["axis"][1][3])
        image = BytesIO()
        fig.savefig(image, format='svg')
        image.seek(0)
        images.append(svg2rlg(image))
    return images


class HealBoneGaitReport:

    def __init__(self, path, summaryData=[], graph=None):
        self.path = path
        self.summaryData = summaryData
        self.graph = graph
        self.styleSheet = getSampleStyleSheet()
        self.elements = []
        pdfmetrics.registerFont(TTFont('SimSun', resourcePath + 'font/SimSun.ttf'))
        pdfmetrics.registerFont(TTFont('msyh', resourcePath + 'font/msyh.ttc'))

        # 报告封面
        self.coverPage()

        # 汇总页
        self.summaryPage()

        # 图表页
        self.graphPage()

        self.doc = SimpleDocTemplate(self.path, pagesize=LETTER)

    def coverPage(self):
        img = Image(resourcePath + 'static/healbone_banner.png', kind='proportional')
        img.drawHeight = 0.5 * inch
        img.drawWidth = 2.6 * inch
        img.hAlign = 'LEFT'
        self.elements.append(img)

        self.elements.append(HeightSpacer(100))

        img = Image(resourcePath + 'static/gait-bold.png')
        img.drawHeight = 2.5 * inch
        img.drawWidth = 6.5 * inch
        self.elements.append(img)

        self.elements.append(HeightSpacer(250))

        psDetalle = ParagraphStyle('Resumen', fontSize=9, leading=14, justifyBreaks=1, alignment=TA_LEFT, justifyLastLine=1)
        time_fmt_str = time.asctime(time.localtime(time.time()))
        text = """HEALBONE GAIT ANALYSIS REPORT<br/>
        PATIENT: QianJin Tang<br/>
        REPORT TIME: """ + time_fmt_str + """<br/>
        LAB: NanJin HealBone Lab1<br/>
        """
        paragraphReportSummary = Paragraph(text, psDetalle)
        self.elements.append(paragraphReportSummary)

    def summaryPage(self):
        self.elements.append(PageBreak())

        self.elements.append(ParagraphReportHeader(fontSize=16, text='报告详情'))
        self.elements.append(HeightSpacer())
        self.elements.append(ReportDivider())
        self.elements.append(HeightSpacer())

        # self.elements.append(ParagraphReportHeader(text='主要参数', color=colorBlue0))
        self.elements.append(HeightSpacer(heightPixels=10))

        """
        主要参数
        """

        self.elements.append(ReportNoBorderTable(lineData=self.summaryData, colWidths=120,
                                                 rowHeights=28, tableStyle=[
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ("ALIGN", (1, 0), (1, -1), 'RIGHT'),
                ("ALIGN", (2, 0), (2, -1), 'CENTER'),
                ("ALIGN", (3, 0), (3, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, 0), colorGreen2),
                ('BACKGROUND', (0, -1), (-1, -1), colorBlue1),
                ('LINEABOVE', (0, 0), (-1, -1), 1, colorBlue1),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONT', (0, 0), (-1, -1), 'msyh'),
                ('FONTSIZE', (0, 0), (-1, -1), 14)
            ]))
        self.elements.append(HeightSpacer(heightPixels=10))
        self.elements.append(ParagraphReportHeader(text='*注: HealBone Lab的Gait检测结果仅对步态评估提供参考建议', color=colorBlack, fontSize=12))

    def graphPage(self):
        drawings = generate_angle_plots(self.graph)
        for drawing in drawings:
            self.elements.append(PageBreak())
            self.elements.append(drawing)

    def exportPDF(self):
        # Build
        self.doc.multiBuild(self.elements, canvasmaker=PageHeaderFooter)


if __name__ == '__main__':
    report = HealBoneGaitReport('report-example.pdf', summaryData=[])
