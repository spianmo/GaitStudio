import copy

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QSlider, QComboBox, QLineEdit, QLabel, QSpinBox

from Dialog import Ui_Dialog
from evaluate.EvaluateCore import InfoForm, RequireCollect


class ViewModel:
    pass


class QRequireCollectDialog(QDialog, Ui_Dialog):
    def __init__(self, parent=None, metadata=None):
        super(QRequireCollectDialog, self).__init__(parent)
        if metadata is None:
            metadata = {}
        self.setupUi(self)
        self.viewModel = {
            RequireCollect.gender: "Male",
            RequireCollect.age: 25,
            RequireCollect.side: "left",
            RequireCollect.eyesClosed: True,
            RequireCollect.name: "",
            RequireCollect.time: 15
        }
        self.labels = {}
        for formItemMeta in metadata:

            formMeta = InfoForm[formItemMeta]
            self.formLayout.setHorizontalSpacing(10)
            self.formLayout.setContentsMargins(10, 10, 10, 10)
            if formMeta["type"] == "slider":
                sliderAbs = QSlider()
                sliderAbs.__setattr__("formKey", formItemMeta)
                sliderAbs.__setattr__("format", formMeta["title"])
                sliderAbs.setMinimum(formMeta["range"][0])
                sliderAbs.setMaximum(formMeta["range"][1])
                sliderAbs.setPageStep(formMeta["step"])
                sliderAbs.setOrientation(Qt.Horizontal)
                sliderAbs.setValue(formMeta["defaultValue"])
                sliderAbs.valueChanged.connect(
                    lambda value: self.valueChange(sliderAbs.__getattribute__("formKey"), value,
                                                   sliderAbs.__getattribute__("format")))
                self.labels[sliderAbs.__getattribute__("formKey")] = QLabel(
                    formMeta["title"].replace("{}",
                                              str(formMeta[
                                                      "defaultValue"]) if "defaultValue" in formMeta.keys() else ""))
                self.formLayout.addRow(self.labels[sliderAbs.__getattribute__("formKey")], sliderAbs)
            elif formMeta["type"] == "select":
                comboxAbs = QComboBox()
                comboxAbs.__setattr__("formKey", formItemMeta)
                comboxAbs.__setattr__("format", formMeta["title"])
                comboxAbs.__setattr__("_item", formMeta["_item"])
                comboxAbs.setEnabled(True)
                comboxAbs.setCurrentIndex(-1)
                for comboIndex, comboItem in enumerate(formMeta["item"]):
                    comboxAbs.addItem(comboItem)
                comboxAbs.currentIndexChanged.connect(
                    lambda value: self.valueChange(comboxAbs.__getattribute__("formKey"),
                                                   comboxAbs.__getattribute__("_item")[value],
                                                   comboxAbs.__getattribute__("format")))
                self.labels[comboxAbs.__getattribute__("formKey")] = QLabel(
                    formMeta["title"].replace("{}",
                                              str(formMeta[
                                                      "defaultValue"]) if "defaultValue" in formMeta.keys() else ""))
                self.formLayout.addRow(self.labels[comboxAbs.__getattribute__("formKey")], comboxAbs)
            elif formMeta["type"] == "input":
                inputEdit = QLineEdit()
                inputEdit.__setattr__("formKey", formItemMeta)
                inputEdit.__setattr__("format", formMeta["title"])
                inputEdit.setPlaceholderText(f"请填写{formMeta['title']}")
                inputEdit.textEdited.connect(
                    lambda value: self.valueChange(inputEdit.__getattribute__("formKey"), value,
                                                   inputEdit.__getattribute__("format")))
                self.labels[inputEdit.__getattribute__("formKey")] = QLabel(
                    formMeta["title"].replace("{}",
                                              str(formMeta[
                                                      "defaultValue"]) if "defaultValue" in formMeta.keys() else ""))
                self.formLayout.addRow(self.labels[inputEdit.__getattribute__("formKey")], inputEdit)
            elif formMeta["type"] == "spinbox":
                spinbox = QSpinBox()
                spinbox.setValue(formMeta["defaultValue"])
                spinbox.__setattr__("formKey", formItemMeta)
                spinbox.__setattr__("format", formMeta["title"])
                spinbox.valueChanged.connect(
                    lambda value: self.valueChange(spinbox.__getattribute__("formKey"), value,
                                                   spinbox.__getattribute__("format")))
                self.labels[spinbox.__getattribute__("formKey")] = QLabel(
                    formMeta["title"].replace("{}",
                                              str(formMeta[
                                                      "defaultValue"]) if "defaultValue" in formMeta.keys() else ""))
                self.formLayout.addRow(self.labels[spinbox.__getattribute__("formKey")], spinbox)

    def valueChange(self, key, value, formatStr):
        if "{}" in formatStr:
            self.labels[RequireCollect(key)].setText(formatStr.format(value))
        self.viewModel[RequireCollect(key)] = value

    def getResult(self):
        return self.viewModel
