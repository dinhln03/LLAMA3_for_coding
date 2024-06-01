import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QAction, QMessageBox, QStatusBar

from PyMailConfigWindow import ConfigWindow
from PyMailReceiverModel import ReceiverModel
from PyMailReceiverView import ReceiverView
from PyMailSenderModel import SenderModel
from PyMailSenderWindow import SenderWindow
from PyMailSplitWidget import SplitWidget
from PyMailStartUpWindow import StartUpWindow
from PyMailToolBar import ToolBar


class PyMailMainWindow(QMainWindow):
    def __init__(self, delegate):
        super().__init__()
        self.setWindowTitle("PyMail")
        self.setWindowIcon(QIcon(r"res\logo.png"))
        self.setCentralWidget(SplitWidget(self))
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        self.setupUI()
        self.show()
        self.addToolBar(ToolBar(self))
        self.delegate = delegate
        self.delegate.registerView(self)
        self.setStatusBar(QStatusBar())
        self.statusBar()
        self.setStatusTip("Ready")
        self.startUpWindow = StartUpWindow(self, self.delegate)

    def setupUI(self):
        self.setupMenuBar()

    def setupMenuBar(self):
        menuBar = self.menuBar()
        self.setupFileMenu(menuBar)
        self.setupEditMenu(menuBar)
        self.setupOptionsMenu(menuBar)
        self.setupHelpMenu(menuBar)

    def setupFileMenu(self, menuBar):
        fileMenu = menuBar.addMenu("File")
        self.setFileMenuActions(fileMenu)

    def setupEditMenu(self, menuBar):
        editMenu = menuBar.addMenu("Edit")

    def setupOptionsMenu(self, menuBar):
        optionsMenu = menuBar.addMenu("Options")
        settingsAction = QAction(QIcon(r"res\settings.png"), "Settings", optionsMenu)
        settingsAction.setStatusTip("Settings")
        settingsAction.triggered.connect(self.showSettings)
        optionsMenu.addAction(settingsAction)

    def setupHelpMenu(self, menuBar):
        helpMenu = menuBar.addMenu("Help")

    def setFileMenuActions(self, fileMenu):
        exitAction = QAction(QIcon(r"res\exit.png"), "Exit", fileMenu)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

    def showSettings(self):
        settingsView = ConfigWindow(self)
        self.delegate.reset()
        self.delegate.configView = settingsView
        self.centralWidget().changeRightWidget(settingsView)

    def showHelp(self):
        pass

    def receiveMail(self):
        self.delegate.reset()
        receiverView = ReceiverView()
        self.delegate.receiverView = receiverView
        receiverModel = ReceiverModel()
        receiverModel.delegate = self.delegate
        self.delegate.receiverModel = receiverModel
        receiverView.delegate = self.delegate
        self.centralWidget().changeLeftWidget(receiverView)

    def showNewMail(self):
        newMailView = SenderWindow()
        newMailModel = SenderModel()
        self.delegate.reset()
        self.delegate.senderView = newMailView
        self.delegate.senderModel = newMailModel
        newMailView.delegate = self.delegate
        newMailModel.delegate = self.delegate
        newMailView.set_actions()
        self.centralWidget().changeRightWidget(newMailView)

    def closeEvent(self, event):
        event.ignore()
        self.exit()

    def resizeEvent(self, event):
        self.centralWidget().resizeWidget()

    def exit(self):
        msg = QMessageBox.question(None, "Exit PyMail", "Do You want to quit")
        if msg == QMessageBox.Yes:
            self.destroy()
            sys.exit()