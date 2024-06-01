import socket, threading, sys, traceback, os, tkinter

from ui import Ui_MainWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox, Tk
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from RtpPacket import RtpPacket

RECV_SIZE = 20480 + 14
HIGHT = 500

CACHE_FILE_NAME = "cache-"
CACHE_FILE_EXT = ".jpg"

class Client:
    INIT = 0
    READY = 1
    PLAYING = 2
    state = INIT
    
    SETUP = 0
    PLAY = 1
    PAUSE = 2
    TEARDOWN = 3
    FASTER = 4
    SLOWER = 5
    
    # Initiation..
    def __init__(self, serveraddr, serverport, rtpport, filename):
        self.page_main = Ui_MainWindow()
        self.state == self.READY
        self.serverAddr = serveraddr
        self.serverPort = int(serverport)
        self.rtpPort = int(rtpport)
        self.fileName = filename
        self.rtspSeq = 0
        self.sessionId = 0
        self.requestSent = -1
        self.teardownAcked = 0
        self.connectToServer()
        self.frameNbr = 0
        self.createWidgets()
        

    def createWidgets(self):
        app = QtWidgets.QApplication(sys.argv)
        page_tmp = QtWidgets.QMainWindow()
        self.page_main.setupUi(page_tmp)
        page_tmp.show()

        self.page_main.btn_setup.clicked.connect(lambda: self.setupMovie())
        self.page_main.btn_play.clicked.connect(lambda: self.playMovie())
        self.page_main.btn_pause.clicked.connect(lambda: self.pauseMovie())
        self.page_main.btn_teardown.clicked.connect(lambda: self.exitClient())
        self.page_main.btn_faster.clicked.connect(lambda: self.fasterMovie())
        self.page_main.btn_slower.clicked.connect(lambda: self.slowerMovie())


        sys.exit(app.exec_())

    def fasterMovie(self):
        """Let movie faster."""
        if self.state == self.PLAYING or self.state == self.READY:
            self.sendRtspRequest(self.FASTER)

    def slowerMovie(self):
        """Let movie slower."""
        if self.state == self.PLAYING or self.state == self.READY:
            self.sendRtspRequest(self.SLOWER)

    def setupMovie(self):
        """Setup init."""
        if self.state == self.INIT:
            self.sendRtspRequest(self.SETUP)
    
    def exitClient(self):
        """Teardown the client."""
        self.sendRtspRequest(self.TEARDOWN)        
        sys.exit(0) # Close the gui window
        print(os.remove(CACHE_FILE_NAME + str(self.sessionId) + CACHE_FILE_EXT)) # Delete the cache image from video

    def pauseMovie(self):
        """Pause movie."""
        if self.state == self.PLAYING:
            self.sendRtspRequest(self.PAUSE)
    
    def playMovie(self):
        """Play movie."""
        if self.state == self.READY:
            # Create a new thread to listen for RTP packets
            threading.Thread(target=self.listenRtp).start()
            self.playEvent = threading.Event()
            self.playEvent.clear()
            self.sendRtspRequest(self.PLAY)
    
    def listenRtp(self):        
        """Listen for RTP packets."""
        while 1:
            try:
                cachename = CACHE_FILE_NAME + str(self.sessionId) + CACHE_FILE_EXT
                file = open(cachename, "wb+")
                while 1:
                    data = self.rtpSocket.recv(RECV_SIZE)
                    if data:
                        rtpPacket = RtpPacket()
                        rtpPacket.decode(data)

                        # self.cutFrameList.append(rtpPacket.getPayload())
                        
                        currFrameNbr = rtpPacket.seqNum()
                        file.write(rtpPacket.getPayload())
                        print("Current Seq Num: " + str(currFrameNbr))
                        
                        if currFrameNbr > self.frameNbr and rtpPacket.getIfEnd(): # Discard the late packet
                            self.frameNbr = currFrameNbr
                            self.updateMovie(cachename)
                            file.close()
                            break
            except:
                # Stop listening upon requesting PAUSE or TEARDOWN
                if self.playEvent.isSet(): 
                    break

                print('Frame receiving failed!')

                # Upon receiving ACK for TEARDOWN request,
                # close the RTP socket
                if self.teardownAcked == 1:
                    self.rtpSocket.shutdown(socket.SHUT_RDWR)
                    self.rtpSocket.close()
                    break
                    
    def writeFrame(self):
        """Write the received frame to a temp image file. Return the image file."""
        cachename = CACHE_FILE_NAME + str(self.sessionId) + CACHE_FILE_EXT
        file = open(cachename, "wb")
        for item in self.cutFrameList:
            file.write(item)
        file.close()
        
        return cachename
    
    def updateMovie(self, imageFile):
        """Update the image file as video frame in the GUI."""
        pixmap = QtGui.QPixmap(imageFile)
        self.page_main.label_display.setPixmap(pixmap)
        self.page_main.label_display.setScaledContents(True)
        
    def connectToServer(self):
        """Connect to the Server. Start a new RTSP/TCP session."""
        self.rtspSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.rtspSocket.connect((self.serverAddr, self.serverPort))
        except:
            # tkMessageBox.showwarning('Connection Failed', 'Connection to \'%s\' failed.' %self.serverAddr)
            messagebox.showwarning('Connection Failed', 'Connection to \'%s\' failed.' %self.serverAddr)
    
    def sendRtspRequest(self, requestCode):
        """Send RTSP request to the server."""
        
        # Setup
        if requestCode == self.SETUP and self.state == self.INIT:
            threading.Thread(target=self.recvRtspReply).start()
            # Update RTSP sequence number.
            self.rtspSeq += 1
            
            # Write the RTSP request to be sent.
            request = 'SETUP ' + self.fileName + ' RTSP/1.0\nCSeq: ' + str(self.rtspSeq) + '\nTransport: RTP/UDP; client_port= ' + str(self.rtpPort)
            
            # Keep track of the sent request.
            self.requestSent = self.SETUP 
        
        # Play
        elif requestCode == self.PLAY and self.state == self.READY:
            self.rtspSeq += 1
            request = 'PLAY ' + self.fileName + ' RTSP/1.0\nCSeq: ' + str(self.rtspSeq) + '\nSession: ' + str(self.sessionId)
            self.requestSent = self.PLAY
        
        # Pause
        elif requestCode == self.PAUSE and self.state == self.PLAYING:
            self.rtspSeq += 1
            request = 'PAUSE ' + self.fileName + ' RTSP/1.0\nCSeq: ' + str(self.rtspSeq) + '\nSession: ' + str(self.sessionId)
            self.requestSent = self.PAUSE
            
        # Teardown
        elif requestCode == self.TEARDOWN and not self.state == self.INIT:
            self.rtspSeq += 1
            request = 'TEARDOWN ' + self.fileName + ' RTSP/1.0\nCSeq: ' + str(self.rtspSeq) + '\nSession: ' + str(self.sessionId) 
            self.requestSent = self.TEARDOWN
        
        # Faster
        elif requestCode == self.FASTER and (self.state == self.PLAYING or self.state == self.READY):
            self.rtspSeq += 1
            request = 'FASTER ' + self.fileName + ' RTSP/1.0\nCSeq: ' + str(self.rtspSeq) + '\nSession: ' + str(self.sessionId) 
        
        # Slower
        elif requestCode == self.SLOWER and (self.state == self.PLAYING or self.state == self.READY):
            self.rtspSeq += 1
            request = 'SLOWER ' + self.fileName + ' RTSP/1.0\nCSeq: ' + str(self.rtspSeq) + '\nSession: ' + str(self.sessionId) 

        else:
            return
        
        # Send the RTSP request using rtspSocket.
        self.rtspSocket.send(request.encode())
        
        print('\nData sent:\n' + request)
    
    def recvRtspReply(self):
        """Receive RTSP reply from the server."""
        while True:
            reply = self.rtspSocket.recv(1024)
            
            if reply: 
                self.parseRtspReply(reply.decode("utf-8"))
            
            # Close the RTSP socket upon requesting Teardown
            if self.requestSent == self.TEARDOWN:
                self.rtspSocket.shutdown(socket.SHUT_RDWR)
                self.rtspSocket.close()
                break
    
    def parseRtspReply(self, data):
        """Parse the RTSP reply from the server."""
        lines = str(data).split('\n')
        seqNum = int(lines[1].split(' ')[1])
        
        # Process only if the server reply's sequence number is the same as the request's
        if seqNum == self.rtspSeq:
            session = int(lines[2].split(' ')[1])
            # New RTSP session ID
            if self.sessionId == 0:
                self.sessionId = session
            
            # Process only if the session ID is the same
            if self.sessionId == session:
                if int(lines[0].split(' ')[1]) == 200: 
                    if self.requestSent == self.SETUP:
                        # Update RTSP state.
                        self.state = self.READY
                        # Open RTP port.
                        self.openRtpPort()
                    elif self.requestSent == self.PLAY:
                        self.state = self.PLAYING
                    elif self.requestSent == self.PAUSE:
                        self.state = self.READY
                        # The play thread exits. A new thread is created on resume.
                        self.playEvent.set()
                    elif self.requestSent == self.TEARDOWN:
                        self.state = self.INIT
                        # Flag the teardownAcked to close the socket.
                        self.teardownAcked = 1 
    
    def openRtpPort(self):
        """Open RTP socket binded to a specified port."""
        # Create a new datagram socket to receive RTP packets from the server
        self.rtpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Set the timeout value of the socket to 0.5sec
        self.rtpSocket.settimeout(0.5)
        
        try:
            # Bind the socket to the address using the RTP port given by the client user
            self.rtpSocket.bind(("", self.rtpPort))
        except:
            messagebox.showwarning('Unable to Bind', 'Unable to bind PORT=%d' %self.rtpPort)

    def handler(self):
        """Handler on explicitly closing the GUI window."""
        self.pauseMovie()
        if messagebox.askokcancel("Quit?", "Are you sure you want to quit?"):
            self.exitClient()
        else: # When the user presses cancel, resume playing.
            self.playMovie()
    
if __name__ == "__main__": 
    try:
        # serverAddr = sys.argv[1]
        # serverPort = sys.argv[2]
        # rtpPort = sys.argv[3]
        # fileName = sys.argv[4]    

        serverAddr = sys.argv[1]
        serverPort = sys.argv[4]
        rtpPort = sys.argv[3]
        fileName = sys.argv[2]    
    except:
        print ("[Usage: ClientLauncher.py Server_name Server_port RTP_port Video_file]\n")    

    # root = tkinter.Tk()

    client = Client(serverAddr, serverPort, rtpPort, fileName)
    # client.master.title('RTP Client')
    # root.mainloop()