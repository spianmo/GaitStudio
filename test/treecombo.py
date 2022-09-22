from PySide2.QtGui import *
from PySide2.QtWidgets import *


class TreeComboBox(QComboBox):
    def __init__(self, *args):
        super().__init__(*args)
        tree_view = QTreeView(self)
        tree_view.setHeaderHidden(True)
        tree_view.setFrameShape(QFrame.NoFrame)
        tree_view.setEditTriggers(tree_view.NoEditTriggers)
        tree_view.setAlternatingRowColors(True)
        tree_view.setSelectionBehavior(tree_view.SelectRows)
        tree_view.setWordWrap(True)
        tree_view.setAllColumnsShowFocus(True)
        self.setView(tree_view)


app = QApplication([])

combo = TreeComboBox()

parent_item = QStandardItem(QIcon("../resources/smallscreen.png"), '膝关节')
parent_item.setSelectable(False)
parent_item.appendRow([QStandardItem('下蹲')])

model = QStandardItemModel()
model.appendRow([parent_item])
model.appendRow([QStandardItem(QIcon("../resources/smallscreen.png"), '髋关节')])
model.setHeaderData(0, Qt.Horizontal, '评估选项', Qt.DisplayRole)
combo.setModel(model)

combo.show()
app.exec_()
