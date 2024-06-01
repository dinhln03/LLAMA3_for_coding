import RPi.GPIO as gpio
import time

class Tank:

    def __init__(self, name):
        self.name = name

    def init(self):
        gpio.setmode(gpio.BCM)
        gpio.setup(17, gpio.OUT)
        gpio.setup(22, gpio.OUT)
        gpio.setup(23, gpio.OUT)
        gpio.setup(24, gpio.OUT)


    def forward(self):
        self.init()
        gpio.output(17, True)  #M1 FWD
        gpio.output(22, False) #M1 REV
        gpio.output(23, True)  #M2 FWD
        gpio.output(24, False) #M2 REV
        #time.sleep(sec)
        #gpio.cleanup()


    def reverse(self, sec):
        self.init()
        gpio.output(17, False)
        gpio.output(22, True)
        gpio.output(23, False)
        gpio.output(24, True)
        time.sleep(sec)
        gpio.cleanup()


    def left(self, sec):
        self.init()
        gpio.output(17, False)
        gpio.output(22, True)
        gpio.output(23, False)
        gpio.output(24, False)
        time.sleep(sec)
        gpio.cleanup()


    def right(self, sec):
        self.init()
        gpio.output(17, False)
        gpio.output(22, False)
        gpio.output(23, False)
        gpio.output(24, True)
        time.sleep(sec)
        gpio.cleanup()
        
    def stop(self):
        self.init()
        gpio.output(17, False)
        gpio.output(22, False)
        gpio.output(23, False)
        gpio.output(24, False)
        gpio.cleanup()

    def init_test(self):
        self.forward(.05)
        time.sleep(.1)
        self.reverse(.05)
        time.sleep(.1)
        self.left(.05)
        time.sleep(.1)
        self.right(.05)
        print(f"Initialization Test Passed! {self.name} is ready to roll!")
        

