
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox
from PyQt5 import uic
from os.path import join, dirname, abspath
from qtpy.QtCore import Slot, QTimer, QThread, Signal, QObject, Qt
#from PyQt5 import Qt

_ST_DLG = join(dirname(abspath(__file__)), 'startdialog.ui')

class StartDialog(QDialog):
    def __init__(self, parent):
        super(StartDialog, self).__init__() # Call the inherited classes __init__ method
        #super().__init__(parent)
        uic.loadUi(_ST_DLG, self)
        self.hideText()
        self.index = 0
        self.labels = [self.label01, self.label02, self.label03, self.label04, self.label05, self.label06]
        self.timer = QTimer()
        self.timer.timeout.connect(self.serialText)
        self.timer.start(1060)
        self.setWindowModality(Qt.ApplicationModal)
        self.exec_()
    
    @Slot()
    def on_ok_clicked(self):
        self.timer.stop()
        self.close()

    def hideText(self):
        self.label01.hide()
        self.label02.hide()
        self.label03.hide()
        self.label04.hide()
        self.label05.hide()
        self.label06.hide()

    def serialText(self):
        self.labels[self.index].show()
        if self.index < 5:
            self.index += 1
        else:
            self.timer.stop()