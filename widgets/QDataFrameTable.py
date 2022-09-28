import typing

import PySide2
import pandas as pd
from PySide2 import QtCore, QtGui
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class PandasModel(QAbstractTableModel):

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe
        self._colors = ['#15169a', '#21168e', '#2d1682', '#44166b', '#491666', '#54165b', '#63164c', '#75163a',
                        '#83162c', '#91161e', '#95161a']

    def rowCount(self, parent=QModelIndex()) -> int:
        if parent == QModelIndex():
            return len(self._dataframe)

        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        if parent == QModelIndex():
            return len(self._dataframe.columns) - 1
        return 0

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns[section + 1])

            if orientation == Qt.Vertical:
                return str(self._dataframe.index[section])

        return None

    def data(self, index: PySide2.QtCore.QModelIndex, role: int = ...) -> typing.Any:
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            return str(round(self._dataframe.iloc[index.row(), index.column() + 1], 8))

        if role == Qt.BackgroundRole:
            value = self._dataframe.iloc[index.row()][index.column() + 1]
            value = int(value)
            # return QColor(self._colors[round(value / 180 * (len(self._colors) - 1))])
            rangedValue = value / 180
            return QColor(round(255 * rangedValue), 0, round(255 * (1 - rangedValue)), 130)

        return None

    def refresh(self):
        self.layoutChanged.emit()


class DataFrameTable(QTableWidget):
    def __init__(self, df, *args):
        super().__init__(*args)
        self.setColumnCount(len(df.columns))
        headerTitles = df.columns.to_list()
        self.setHorizontalHeaderLabels(headerTitles)
        self.horizontalHeader().setDragEnabled(True)
        # self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
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
        self.setColumnCount(table_col - 1)
        for row in range(table_row):
            for col in range(table_col):
                if col == table_col - 1:
                    continue
                item = QTableWidgetItem(str(round(self.df.iloc[row, col + 1], 8)))
                self.setItem(row, col, item)
