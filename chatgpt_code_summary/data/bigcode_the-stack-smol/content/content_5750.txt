from machine import Pin
import utime
led = Pin(28, Pin.OUT)
onboard_led = Pin(25, Pin.OUT)
led.low()
onboard_led.high()
while True:
    led.toggle()
    onboard_led.toggle()
    print("Toggle")
    utime.sleep(0.5)
    