from .protocol import *
import serial
import time
from threading import Lock
from PIL import Image, ImageDraw, ImageOps, ImageFont

class Device:
    ser = None
    inbuffer = ""

    awaitingResponseLock = Lock()

    testmode = False
    nLeds = 0
    dispW = 0
    dispH = 0
    rotFactor = 0
    rotCircleSteps = 0

    bannerHeight = 12 #Defines the height of top and bottom banner

    imageBuffer = []

    callbacks = {} #This object stores callback functions that react directly to a keypress reported via serial

    ledState = None         #Current LED status, so we can animate them over time
    ledTime = None          #Last time LEDs were set

    debug = True

    def connect(self, dev):
        print("Connecting to ", dev, ".")
        self.ser = serial.Serial(dev, 115200, timeout=1)
        if not self.requestInfo(3):
            self.disconnect()
            return False
        if self.testmode:
            print("Connection to ", self.ser.name, " was successfull, but the device is running the hardware test firmware, which cannot be used for anything but testing. Please flash the proper inkkeys firmware to use it.")
            return False
        print("Connected to ", self.ser.name, ".")
        return True

    def disconnect(self):
        if self.ser != None:
            self.ser.close()
            self.ser = None

    def sendToDevice(self, command):
        if self.debug:
            print("Sending: " + command)
        self.ser.write((command + "\n").encode())

    def sendBinaryToDevice(self, data):
        if self.debug:
            print("Sending " + str(len(data)) + " bytes of binary data.")
        self.ser.write(data)

    def readFromDevice(self):
        if self.ser.in_waiting > 0:
            self.inbuffer += self.ser.read(self.ser.in_waiting).decode().replace("\r", "")
        chunks = self.inbuffer.split("\n", 1)
        if len(chunks) > 1:
            cmd = chunks[0]
            self.inbuffer = chunks[1]
            if self.debug:
                print("Received: " + cmd)
            return cmd
        return None

    def poll(self):
        with self.awaitingResponseLock:
            input = self.readFromDevice()
        if input != None:
            if input[0] == KeyCode.JOG.value and (input[1:].isdecimal() or (input[1] == '-' and input[2:].isdecimal())):
                if KeyCode.JOG.value in self.callbacks:
                    self.callbacks[KeyCode.JOG.value](int(input[1:]))
            elif input in self.callbacks:
                self.callbacks[input]()

    def registerCallback(self, cb, key):
        self.callbacks[key.value] = cb

    def clearCallback(self, key):
        if key.value in self.callbacks:
            del self.callbacks[key.value]

    def clearCallbacks(self):
        self.callbacks = {}

    def assignKey(self, key, sequence):
        self.sendToDevice(CommandCode.ASSIGN.value + " " + key.value + (" " + " ".join(sequence) if len(sequence) > 0 else ""))

    def sendLed(self, colors):
        self.sendToDevice(CommandCode.LED.value + " " + " ".join(colors))

    def requestInfo(self, timeout):
        with self.awaitingResponseLock:
            print("Requesting device info...")
            start = time.time()
            self.sendToDevice(CommandCode.INFO.value)
            line = self.readFromDevice()
            while line != "Inkkeys":
                if time.time() - start > timeout:
                    return False
                if line == None:
                    time.sleep(0.1)
                    line = self.readFromDevice()
                    continue
                print("Skipping: ", line)
                line = self.readFromDevice()
            print("Header found. Waiting for infos...")
            line = self.readFromDevice()
            while line != "Done":
                if time.time() - start > timeout:
                    return False
                if line == None:
                    time.sleep(0.1)
                    line = self.readFromDevice()
                    continue
                if line.startswith("TEST "):
                    self.testmode = line[5] != "0"
                elif line.startswith("N_LED "):
                    self.nLeds = int(line[6:])
                elif line.startswith("DISP_W "):
                    self.dispW = int(line[7:])
                elif line.startswith("DISP_H "):
                    self.dispH = int(line[7:])
                elif line.startswith("ROT_CIRCLE_STEPS "):
                    self.rotCircleSteps = int(line[17:])
                else:
                    print("Skipping: ", line)
                line = self.readFromDevice()
            print("End of info received.")
            print("Testmode: ", self.testmode)
            print("Number of LEDs: ", self.nLeds)
            print("Display width: ", self.dispW)
            print("Display height: ", self.dispH)
            print("Rotation circle steps: ", self.rotCircleSteps)
            return True

    def sendImage(self, x, y, image):
        self.imageBuffer.append({"x": x, "y": y, "image": image.copy()})
        w, h = image.size
        data = image.convert("1").rotate(180).tobytes()
        self.sendToDevice(CommandCode.DISPLAY.value + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h))
        self.sendBinaryToDevice(data)
        return True

    def resendImageData(self):
        for part in self.imageBuffer:
            image = part["image"]
            x = part["x"]
            y = part["y"]
            w, h = image.size
            data = image.convert("1").rotate(180).tobytes()
            self.sendToDevice(CommandCode.DISPLAY.value + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h))
            self.sendBinaryToDevice(data)
        self.imageBuffer = []

    def updateDisplay(self, fullRefresh=False, timeout=5):
        with self.awaitingResponseLock:
            start = time.time()
            self.sendToDevice(CommandCode.REFRESH.value + " " + (RefreshTypeCode.FULL.value if fullRefresh else RefreshTypeCode.PARTIAL.value))
            line = self.readFromDevice()
            while line != "ok":
                if time.time() - start > timeout:
                    return False
                if line == None:
                    time.sleep(0.1)
                    line = self.readFromDevice()
                    continue
                line = self.readFromDevice()
            self.resendImageData()
            self.sendToDevice(CommandCode.REFRESH.value + " " + RefreshTypeCode.OFF.value)
            line = self.readFromDevice()
            while line != "ok":
                if time.time() - start > timeout:
                    return False
                if line == None:
                    time.sleep(0.1)
                    line = self.readFromDevice()
                    continue
                line = self.readFromDevice()

    def getAreaFor(self, function):
        if function == "title":
            return (0, self.dispH-self.bannerHeight, self.dispW, self.bannerHeight)
        elif function == 1:
            return (0, 0, self.dispW, self.bannerHeight)
        elif function <= 5:
            return (self.dispW//2, (5-function)*self.dispH//4+self.bannerHeight, self.dispW//2, self.dispH//4-2*self.bannerHeight)
        else:
            return (0, (9-function)*self.dispH//4+self.bannerHeight, self.dispW//2, self.dispH//4-2*self.bannerHeight)

    def sendImageFor(self, function, image):
        x, y, w, h = self.getAreaFor(function)
        if (w, h) != image.size:
            if self.debug:
                print("Rescaling image from " + str(image.size) + " to " + str((w, h)) + ".")
            image = image.resize((w, h))
        self.sendImage(x, y, image)

    def sendTextFor(self, function, text=""):
        self.sendToDevice("T "+str(function)+" "+str(text))
    
    def sendIconFor(self, function, icon, inverted=False, centered=True, marked=False, crossed=False):
        x, y, w, h = self.getAreaFor(function)
        img = Image.new("1", (w, h), color=(0 if inverted else 1))
        imgIcon = Image.open(icon).convert("RGB")
        if inverted:
            imgIcon = ImageOps.invert(imgIcon)
        wi, hi = imgIcon.size
        if function < 6:
            pos = ((w-wi)//2 if centered else 0, (h - hi)//2)
        else:
            pos = ((w-wi)//2 if centered else (w - wi), (h - hi)//2)
        img.paste(imgIcon, pos)

        if marked:
            imgMarker = Image.open("icons/chevron-compact-right.png" if function < 6 else "icons/chevron-compact-left.png")
            wm, hm = imgMarker.size
            img.paste(imgMarker, (-16,(h - hm)//2) if function < 6 else (w-wm+16,(h - hm)//2), mask=ImageOps.invert(imgMarker.convert("RGB")).convert("1"))

        if crossed:
            d = ImageDraw.Draw(img)
            d.line([pos[0]+5, pos[1]+5, pos[0]+wi-5, pos[1]+hi-5], width=3)
            d.line([pos[0]+5, pos[1]+hi-5, pos[0]+wi-5, pos[1]+5], width=3)

        self.sendImage(x, y, img)

    def setLeds(self, leds):
        ledStr = ['{:06x}'.format(i) for i in leds]
        self.ledTime = time.time()
        self.ledState = leds
        self.sendLed(ledStr)

    def setKeyLedFor(self, led, color):
        self.sendToDevice(CommandCode.KEYLED.value + " " +  str(led)+ " " + str(color))

    def fadeLeds(self):
        if self.ledState == None:
            return
        p = (3.5 - (time.time() - self.ledTime))/0.5 #Stay on for 3 seconds and then fade out over 0.5 seconds
        if p >= 1:
            return
        if p <= 0:
            self.ledState = None
            self.sendLed(["000000" for i in range(self.nLeds)])
            return
        dimmedLeds = [(int((i & 0xff0000) * p) & 0xff0000) | (int((i & 0xff00) * p) & 0xff00) | (int((i & 0xff) * p) & 0xff) for i in self.ledState]
        ledStr = ['{:06x}'.format(i) for i in dimmedLeds]
        self.sendLed(ledStr)

        
        
        