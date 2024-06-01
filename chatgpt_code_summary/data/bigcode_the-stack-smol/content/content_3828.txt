import serial
import pynmea2

# Probando con el pincho usb azul
ser = serial.Serial('/dev/ttyUSB0',4800)

while 1:
    try:
        data = ser.readline().decode('utf-8')
        if(data.startswith("$GPGGA")):
            parse = pynmea2.parse(data)
            print(repr(parse))
    except UnicodeDecodeError:
        continue