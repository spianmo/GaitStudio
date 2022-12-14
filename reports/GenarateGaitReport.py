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
        self.drawImage(resourcePath + "static/healbone_banner.png", self.width - inch * 8 - 5, self.height - 50,
                       width=100, height=20,
                       preserveAspectRatio=True,
                       mask='auto')
        self.drawImage(resourcePath + "static/gait.png", self.width - inch * 2, self.height - 50, width=100, height=30,
                       preserveAspectRatio=True,
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


def ParagraphReportHeader(color=colorGreen0, fontSize=12, text='', leftIndent=-14, alignment=TA_LEFT, leading=12):
    psHeaderText = ParagraphStyle('Hed0', fontName='msyh', fontSize=fontSize, alignment=alignment, borderWidth=3,
                                  textColor=color,
                                  leftIndent=leftIndent, leading=leading)
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


class HealBoneGaitReport:

    def __init__(self, path, patientName, evaluateName, evaluateNameEn, SpatiotemporalData=None, ROMData=None, SpatiotemporalGraph=None,
                 ROMGraph=None):
        self.path = path
        self.patientName = patientName
        self.evaluateName = evaluateName
        self.evaluateNameEn = evaluateNameEn
        self.SpatiotemporalData = SpatiotemporalData
        self.ROMData = ROMData
        self.SpatiotemporalGraph = SpatiotemporalGraph
        self.ROMGraph = ROMGraph
        self.styleSheet = getSampleStyleSheet()
        self.elements = []
        pdfmetrics.registerFont(TTFont('SimSun', resourcePath + 'font/SimSun.ttf'))
        pdfmetrics.registerFont(TTFont('msyh', resourcePath + 'font/msyh.ttc'))

        # ????????????
        self.coverPage()

        # Spatiotemporal???????????????
        self.SpatiotemporalPage()

        # Spatiotemporal?????????
        self.SpatiotemporalGraphPage()

        # ROM??????????????????????????????
        self.ROMPage()

        self.doc = SimpleDocTemplate(self.path, pagesize=LETTER, author="NanJin HealBone Lab",
                                     title=f"HealBone {self.evaluateNameEn} Report")

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

        self.elements.append(HeightSpacer(240))

        psDetalle = ParagraphStyle('Resumen', fontName='msyh',
                                   fontSize=12, leading=15, justifyBreaks=1,
                                   alignment=TA_LEFT, justifyLastLine=1)
        # time_fmt_str = time.asctime(time.localtime(time.time()))
        # text = """HEALBONE GAIT ANALYSIS REPORT<br/>
        # PATIENT: QianJin Tang<br/>
        # REPORT TIME: """ + time_fmt_str + """<br/>
        # LAB: NanJin HealBone Lab1<br/>
        # """
        time_fmt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        text = """HEALBONE """+ self.evaluateNameEn.upper() +""" ANALYSIS REPORT<br/>
        ????????????""" + self.evaluateName + """??????<br/>
        ???????????????: """ + self.patientName + """<br/>
        ????????????: """ + time_fmt_str + """<br/>
        ??????: ???????????????????????????<br/>
        """
        paragraphReportSummary = Paragraph(text, psDetalle)
        self.elements.append(paragraphReportSummary)

    def SpatiotemporalPage(self):
        self.elements.append(PageBreak())

        self.elements.append(ParagraphReportHeader(fontSize=16, text='Spatiotemporal parameters ????????????'))
        self.elements.append(HeightSpacer())
        self.elements.append(ReportDivider())
        self.elements.append(HeightSpacer())

        # self.elements.append(ParagraphReportHeader(text='????????????', color=colorBlue0))
        self.elements.append(HeightSpacer(heightPixels=10))

        """
        ????????????
        """

        self.elements.append(
            ReportNoBorderTable(lineData=self.SpatiotemporalData, colWidths=120, rowHeights=45, tableStyle=[
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ("ALIGN", (1, 0), (1, -1), 'RIGHT'),
                ("ALIGN", (2, 0), (2, -1), 'CENTER'),
                ("ALIGN", (3, 0), (3, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, 0), colorGreen2),
                ('BACKGROUND', (0, -1), (-1, -1), colorBlue1),
                ('LINEABOVE', (0, 0), (-1, -1), 1, colorBlue1),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONT', (0, 0), (-1, -1), 'msyh'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('LEADING', (0, 0), (-1, -1), 18)
            ]))
        self.elements.append(HeightSpacer(heightPixels=10))
        self.elements.append(
            ParagraphReportHeader(text=f'*???: HealBone Lab???{self.evaluateName}??????????????????{self.evaluateNameEn}????????????????????????', color=colorBlack,
                                  fontSize=10))
        self.elements.append(PageBreak())

    def ROMPage(self):
        for rom_index, romItem in enumerate(self.ROMData):

            self.elements.append(ParagraphReportHeader(fontSize=16, text=('Range of Motion ' + romItem["title"])))
            self.elements.append(HeightSpacer())
            self.elements.append(ReportDivider())
            self.elements.append(HeightSpacer())

            # self.elements.append(ParagraphReportHeader(text='????????????', color=colorBlue0))
            self.elements.append(HeightSpacer(heightPixels=10))

            """
            ????????????
            """

            self.elements.append(
                ReportNoBorderTable(lineData=romItem["list"], colWidths=120, rowHeights=38, tableStyle=[
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ("ALIGN", (1, 0), (1, -1), 'RIGHT'),
                    ("ALIGN", (2, 0), (2, -1), 'CENTER'),
                    ("ALIGN", (3, 0), (3, -1), 'CENTER'),
                    ('BACKGROUND', (0, 0), (-1, 0), colorGreen2),
                    ('BACKGROUND', (0, -1), (-1, -1), colorBlue1),
                    ('LINEABOVE', (0, 0), (-1, -1), 1, colorBlue1),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONT', (0, 0), (-1, -1), 'msyh'),
                    ('FONTSIZE', (0, 0), (-1, -1), 12),
                    ('LEADING', (0, 0), (-1, -1), 15)
                ]))

            self.elements.append(HeightSpacer(heightPixels=10))
            self.elements.append(ParagraphReportHeader(
                text='References: Wem ROM by joint Available:https://www.wikem.org/wiki/Range_of_motion_by_joint (accessed 25.10.2021)',
                color=colorBlack, fontSize=10))
            if romItem["title"] == "??????????????????":
                img = Image(resourcePath + 'static/Knee_Motions.png')
                img.drawHeight = 3.2 * inch
                img.drawWidth = 4.8 * inch
                self.elements.append(img)
                self.elements.append(HeightSpacer(heightPixels=10))
                self.elements.append(ParagraphReportHeader(
                    text='???????????????-?????????0?????140?????\n?????????????????????????????????????????????????????????????????????????????????????????????????????????',
                    color=colorBlack, fontSize=12, leading=20))
                self.elements.append(self.ROMGraph[0])
            if romItem["title"] == "??????????????????":
                img = Image(resourcePath + 'static/Hip_Motions.png')
                img.drawHeight = 6.0 * inch
                img.drawWidth = 6.0 * inch
                self.elements.append(ParagraphReportHeader(
                    text='',
                    color=colorBlack, fontSize=10))
                self.elements.append(img)
                self.elements.append(HeightSpacer(heightPixels=10))
                self.elements.append(ParagraphReportHeader(
                    text='??????????????????0?????120???????????????????????0?????15??-30?????\n???????????????????????????????????????????????????????????????????????????????????????????????????\n'
                         '??????????????????0?????45???????????????????????0?????35?????\n?????????????????????????????????????????????????????????????????????????????????????????????????????????\n'
                         '??????????????????0?????35???????????????????????0?????45?????\n?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????\n',
                    color=colorBlack, fontSize=12, leading=20))
                self.elements.append(self.ROMGraph[1])
                self.elements.append(self.ROMGraph[2])
                self.elements.append(self.ROMGraph[3])
            if romItem["title"] == "???????????????":
                img1 = Image(resourcePath + 'static/Pelvis_Motions.png')
                img1.drawHeight = 2 * inch
                img1.drawWidth = 2 * inch
                self.elements.append(ParagraphReportHeader(
                    text='',
                    color=colorBlack, fontSize=10))
                self.elements.append(HeightSpacer(heightPixels=10))
                self.elements.append(ParagraphReportHeader(
                    text='????????????Pelvis Obliquity',
                    color=colorBlack, fontSize=12, leading=20))
                self.elements.append(img1)

                img2 = Image(resourcePath + 'static/Pelvis_rotation.png')
                img2.drawHeight = 2 * inch
                img2.drawWidth = 2 * inch
                self.elements.append(ParagraphReportHeader(
                    text='',
                    color=colorBlack, fontSize=10))
                self.elements.append(HeightSpacer(heightPixels=10))
                self.elements.append(ParagraphReportHeader(
                    text='????????????Pelvis Rotation',
                    color=colorBlack, fontSize=12, leading=20))
                self.elements.append(img2)
                self.elements.append(self.ROMGraph[4])
            self.elements.append(PageBreak())

    def SpatiotemporalGraphPage(self):
        for drawing in self.SpatiotemporalGraph:
            self.elements.append(drawing)

    def exportPDF(self):
        # Build
        self.doc.multiBuild(self.elements, canvasmaker=PageHeaderFooter)
