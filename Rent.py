import sys
from UI import Ui_MainWindow
from model import *
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QMainWindow
from PyQt5.QtGui import QPixmap
import time

class MainWindow_controller(QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton.clicked.connect(self.buttonClicked)
        self.ui.useLine.editingFinished.connect(self.status)
        self.model = model()

    def buttonClicked(self):
        use = self.ui.useLine.text()
        area = self.ui.areaLine.text()
        room = self.ui.roomsLine.text()
        dinning = self.ui.dinningLine.text()
        bath = self.ui.bathroomsLine.text()
        floor = self.ui.floorsLine.text()
        totalFloor = self.ui.totalFloorsLine.text()
        types = self.ui.typeCombo.currentText()
        district = self.ui.districtCombo.currentText()

        validate = True

        if use != '居住' and use != '商業':
            QMessageBox.warning(self, "Alert", "用途輸入有誤!", QMessageBox.Close, QMessageBox.Ok)

        if len(area) == 0 or len(room) == 0 or len(dinning) == 0 or len(bath) == 0 or \
            len(floor) == 0 or len(totalFloor) == 0 or types == '-' or district == '-':
            QMessageBox.warning(self, "Alert", "資料輸入不完整!", QMessageBox.Close, QMessageBox.Ok)
            return

        if int(floor) <= 0 or int(totalFloor) <= 0 or totalFloor < floor:
            QMessageBox.warning(self, "Alert", "樓層輸入有誤!", QMessageBox.Close, QMessageBox.Ok)
            return


        # print(type(msg))
        # self.qthread = ThreadTask()
        # self.qthread.start()
                
        result = self.model.PredictRent(use, area, room, dinning, bath, floor, totalFloor, types, district)

        self.ui.resultLabel.setText(result)
    
    def status(self):
        live = ["-", "獨立套房", "分租套房", "整層住家", "住宅"]
        business = ["-", "店面", "辦公"]

        if self.ui.useLine.text() == '居住':
            self.ui.typeCombo.clear()
            self.ui.typeCombo.addItems(live)

        elif self.ui.useLine.text() == '商業':
            self.ui.typeCombo.clear()
            self.ui.typeCombo.addItems(business)

        if self.ui.useLine.text() == '居住' or self.ui.useLine.text() == '商業':

            self.ui.floorsLine.setEnabled(True)
            self.ui.totalFloorsLine.setEnabled(True)
            self.ui.typeCombo.setEnabled(True)
            self.ui.districtCombo.setEnabled(True)
            self.ui.pushButton.setEnabled(True)   
        else:
            self.ui.floorsLine.setEnabled(False)
            self.ui.totalFloorsLine.setEnabled(False)
            self.ui.typeCombo.setEnabled(False)
            self.ui.districtCombo.setEnabled(False)
            self.ui.pushButton.setEnabled(False)   


