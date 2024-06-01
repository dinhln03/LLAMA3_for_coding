import RPi.GPIO as gpio
import time
from subprocess import Popen, PIPE, call

pin =38 
gpio.setmode(gpio.BOARD)
gpio.setup(pin, gpio.IN, pull_up_down = gpio.PUD_UP)
PRESSED = 0
prev_state = 1
pressed_time = 0.1
skip_song_mode = False
try:
  while True:
    cur_state = gpio.input(pin)
    if cur_state == PRESSED:
        pressed_time += 0.1 
        print "pressed : " + str( pressed_time)
        if pressed_time > 1:
            call(["espeak", "-ven", "shutting down"])
        elif pressed_time == 0.1:
           skip_song_mode = True
        else:
           skip_song_mode = False
    else:
        pressed_time = 0
        if skip_song_mode == True:
            call(["espeak", "-ven", "skip song"])
            skip_song_mode = False
    time.sleep(0.1)
finally:
    gpio.cleanup()
