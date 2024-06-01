from PySide import QtGui, QtCore
import os, subprocess, shutil, re

class animQt(QtGui.QMainWindow):
    def __init__(self):
        super(animQt, self).__init__()

        self.setGeometry(250,250,360,100)
        style = """
        QMainWindow, QMessageBox{
        background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0.264865 rgba(121, 185, 255, 255), stop:1 rgba(0, 126, 255, 255));
        }
        QPushButton{
        background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:1, y2:0, stop:0.448649 rgba(255, 255, 255, 107), stop:0.464865 rgba(0, 0, 0, 15));
        border:1px solid rgb(0, 170, 255);
        padding:5px;
        color:#FFF;
        border-radius:5px;
        }
        QPushButton:hover{
        background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:1, y2:0, stop:0.448649 rgba(0, 0, 0, 15), stop:0.47 rgba(255, 255, 255, 107));
        }
        QCheckBox{
        color:#FFF;
        }
        QLineEdit{
        background-color:rgba(255, 255, 255, 100);
        color:#FFF;
        border:1px solid rgb(0,170,255);
        border-radius:5px;
        padding:3px;
        }
        QLabel{
        color:#FFF;
        }
        QComboBox{
        background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:1, y2:0, stop:0.448649 rgba(255, 255, 255, 107), stop:0.464865 rgba(0, 0, 0, 15));
        color:#FFF;
        padding:5px;
        border:1px solid rgb(0, 170, 255);
        border-radius:5px;
        }
        QComboBox:hover{
        background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:1, y2:0, stop:0.448649 rgba(0, 0, 0, 15), stop:0.47 rgba(255, 255, 255, 107));
        }
        QComboBox::drop-down{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width:25px;
        border-left-width: 1px;
        border-left-style: solid;
        border-top-right-radius: 5px;
        border-bottom-right-radius: 5px;
        border-left-color: rgb(0, 170, 255);
        }
        QComboBox::down-arrow{
        border-image: url("./down-arrow.png");
        height:30px;
        width:30px;
        }
        """
        effect = QtGui.QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(5)
        effect.setOffset(2,2)
        self.setStyleSheet(style)
        self.setWindowTitle("Exe Generator(py2exe)")

        centralWidget = QtGui.QWidget()
        layout = QtGui.QGridLayout(centralWidget)

        self.foldPath = QtGui.QLineEdit(self)

        openBtn = QtGui.QPushButton(self)
        openBtn.setGraphicsEffect(effect)
        openBtn.setText("Select File")
        openBtn.clicked.connect(self.fileBrowser)
        pyPathInit = QtGui.QLabel(self)
        pyPathInit.setText("Select Python Version")
        self.pyPath = QtGui.QComboBox(self)
        self.pyPath.activated.connect(self.changePyPath)
        effect = QtGui.QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(5)
        effect.setOffset(2, 2)
        self.pyPath.setGraphicsEffect(effect)
        self.checkBox = QtGui.QCheckBox(self)
        self.checkBox.setText("Window Mode")
        checkBtn = QtGui.QPushButton(self)
        checkBtn.clicked.connect(self.createSetup)
        checkBtn.setText("Process")
        effect = QtGui.QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(5)
        effect.setOffset(2, 2)
        checkBtn.setGraphicsEffect(effect)

        layout.addWidget(self.foldPath, 0, 0, 1, 2)
        layout.addWidget(openBtn, 0, 2, 1, 1)
        layout.addWidget(pyPathInit, 1, 0, 1, 1)
        layout.addWidget(self.pyPath, 1, 1, 1, 2)
        layout.addWidget(self.checkBox, 2, 0, 1, 2)
        layout.addWidget(checkBtn, 2, 2, 1, 1)

        self.setCentralWidget(centralWidget)
        self.getInstalledPy()

    def fileBrowser(self):
        browse = QtGui.QFileDialog.getOpenFileName(self, "Select File")
        self.foldPath.setText(browse[0])
        self.foldName = os.path.dirname(browse[0])
        self.filePath = browse[0]
        # self.createSetup()

    def changePyPath(self, index):
        self.setPath = self.pyPath.itemText(index)

    def getInstalledPy(self):
        path = "c:/"
        self.pyPath.addItem("Select")
        for each in os.listdir(path):
            if os.path.isdir(path+each):
                if re.search("Python\d", each, re.I):
                    if os.path.exists(path+each+"/python.exe"):
                        # print path+each+"/python.exe"
                        self.pyPath.addItem(path+each+"/python.exe")
        # self.pyPath.addItem("Z:/workspace_mel/dqepy/py27/Scripts/python.exe")

    def createSetup(self):
        try:
            setupFile = self.foldName.replace('\\','/')+"/setup.py"
            with open(setupFile, 'w') as fd:
                if not self.checkBox.isChecked():
                    fd.write("from distutils.core import setup\n")
                    fd.write("import py2exe\n")
                    fd.write("setup(console =['%s'])"%os.path.basename(self.filePath))
                else:
                    fd.write("from distutils.core import setup\n")
                    fd.write("import py2exe\n")
                    fd.write("setup(windows =['%s'])" % os.path.basename(self.filePath))
            self.cmdProcess()
            shutil.rmtree('%s/build'%self.foldName.replace('\\','/'))
            os.rename("dist",os.path.basename(self.filePath).split('.')[0])
            self.displayError(parent=self, m="Process done successfully!!!", t="Process Done")
        except Exception as e:
            self.displayError(parent=self, m="Please Enter all the values\nbefore clicking process button", t="Invalid Values", type=QtGui.QMessageBox.Critical)

    def cmdProcess(self):
        with open("runBatch.bat", 'w') as fd:
            fd.write("@echo off\n")
            fd.write("cd %s\n" % self.foldName)
            fd.write("%s\n"%self.foldName.replace('\\','/').split("/")[0])
            fd.write('%s setup.py py2exe'%self.setPath)
        try:
            subprocess.call("runBatch.bat", 0, None, None, None, None)
        except:
            self.displayError(parent=self, m="Python modules were missing in the Python Interpreter\nPlease make sure you had py2exe module", t="Invalid Python Version", type=QtGui.QMessageBox.Critical)
        os.remove("runBatch.bat")

    def displayError(self, parent, m=None, t="Error found", type=QtGui.QMessageBox.Information, details = ""):
        dError = QtGui.QMessageBox(parent)
        dError.setText(m)
        dError.setWindowTitle(t)
        dError.setIcon(type)
        dError.setStandardButtons(QtGui.QMessageBox.Ok)
        dError.setEscapeButton(QtGui.QMessageBox.Ok)
        if details != "":
            dError.setDetailedText(details)
        dError.show()

if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    gui = animQt()
    gui.show()
    sys.exit(app.exec_())