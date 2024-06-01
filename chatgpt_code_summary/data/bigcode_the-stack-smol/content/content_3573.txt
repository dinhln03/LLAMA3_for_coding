import sys
from PySide2.QtWidgets import QApplication,QWidget,QMenuBar,QPushButton,QVBoxLayout,QMainWindow

from MainWindow import MainWindow

def start():
    app = QApplication(sys.argv)
    app.setApplicationName("My Little ERP")
    mainWindow = MainWindow(app)
    mainWindow.show()
    sys.exit(app.exec_())
