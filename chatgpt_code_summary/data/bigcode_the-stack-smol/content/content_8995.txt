# led_hello.py - blink external LED to test GPIO pins
# (c) BotBook.com - Karvinen, Karvinen, Valtokari

"led_hello.py - light a LED using Raspberry Pi GPIO"
# Copyright 2013 http://Botbook.com */


import time	# <1>
import os	# <2>

def writeFile(filename, contents):	# <3>
	with open(filename, 'w') as f:	# <4>
		f.write(contents)	# <5>

# main

print "Blinking LED on GPIO 27 once..."		# <6>

if not os.path.isfile("/sys/class/gpio/gpio27/direction"):	# <7>
	writeFile("/sys/class/gpio/export", "27")	# <8>

writeFile("/sys/class/gpio/gpio27/direction", "out")	# <9>

writeFile("/sys/class/gpio/gpio27/value", "1")	# <10>
time.sleep(2)	# seconds	# <11>
writeFile("/sys/class/gpio/gpio27/value", "0")	# <12>

