#!/usr/bin/env python
#version 2.1

from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4 import Qt
import PyQt4.Qwt5 as Qwt
from PyQt4.QtCore import pyqtSignal

class control_button_frame(QtGui.QFrame):
    def __init__(self, parent=None, az_el = None):
        super(control_button_frame, self).__init__()
        self.parent = parent
        self.az_el = az_el
        self.initUI()

    def initUI(self):
        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.init_widgets()
        self.connect_signals()

    def init_widgets(self):
        self.MinusTenButton = QtGui.QPushButton(self)
        self.MinusTenButton.setText("-10.0")
        self.MinusTenButton.setMinimumWidth(45)

        self.MinusOneButton = QtGui.QPushButton(self)
        self.MinusOneButton.setText("-1.0")
        self.MinusOneButton.setMinimumWidth(45)

        self.MinusPtOneButton = QtGui.QPushButton(self)
        self.MinusPtOneButton.setText("-0.1")
        self.MinusPtOneButton.setMinimumWidth(45)

        self.PlusPtOneButton = QtGui.QPushButton(self)
        self.PlusPtOneButton.setText("+0.1")
        self.PlusPtOneButton.setMinimumWidth(45)

        self.PlusOneButton = QtGui.QPushButton(self)
        self.PlusOneButton.setText("+1.0")
        self.PlusOneButton.setMinimumWidth(45)

        self.PlusTenButton = QtGui.QPushButton(self)
        self.PlusTenButton.setText("+10.0")
        self.PlusTenButton.setMinimumWidth(45)

        hbox1 = QtGui.QHBoxLayout()
        hbox1.addWidget(self.MinusTenButton)
        hbox1.addWidget(self.MinusOneButton)
        hbox1.addWidget(self.MinusPtOneButton)
        hbox1.addWidget(self.PlusPtOneButton)
        hbox1.addWidget(self.PlusOneButton)
        hbox1.addWidget(self.PlusTenButton)
        self.setLayout(hbox1)

    def connect_signals(self):
        self.PlusPtOneButton.clicked.connect(self.button_clicked) 
        self.PlusOneButton.clicked.connect(self.button_clicked) 
        self.PlusTenButton.clicked.connect(self.button_clicked) 
        self.MinusPtOneButton.clicked.connect(self.button_clicked) 
        self.MinusOneButton.clicked.connect(self.button_clicked) 
        self.MinusTenButton.clicked.connect(self.button_clicked) 

    def button_clicked(self):
        sender = self.sender()
        self.parent.increment_target_angle(self.az_el,float(sender.text()))        
    

