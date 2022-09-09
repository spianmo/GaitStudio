import numpy as np
from PySide2 import QtCore
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from pandas import DataFrame


class DataFrameTable(QTableWidget):
    def __init__(self, df, *args):
        super().__init__(*args)
        self.setColumnCount(len(df.columns))
        headerTitles = df.columns.to_list()
        self.setHorizontalHeaderLabels(headerTitles)
        self.horizontalHeader().setDragEnabled(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.df = df
        size_policy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setSizePolicy(size_policy)
        self.set_data()

    def set_data(self, new_df=None):
        if new_df is not None:
            self.df = new_df
            titles = self.df.columns.to_list()
            self.setHorizontalHeaderLabels(titles[1:])
        table_row = self.df.shape[0]
        table_col = self.df.shape[1]
        self.setRowCount(table_row)
        self.setColumnCount(table_col)
        for row in range(table_row):
            for col in range(table_col):
                if col == 0:
                    continue
                item = QTableWidgetItem(str(round(self.df.iloc[row, col], 8)))
                self.setItem(row, col, item)
