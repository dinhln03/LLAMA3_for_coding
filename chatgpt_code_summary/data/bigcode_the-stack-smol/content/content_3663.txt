#!/usr/bin/env python
"""
	SHUTDOWN.PY
	
	Shutdown Plugin
	
	(C) 2015, rGunti
"""
import dot3k.lcd as lcd
import dot3k.backlight as backlight
import time, datetime, copy, math, psutil
import sys
import os
from dot3k.menu import Menu, MenuOption

class Shutdown(MenuOption):
	def __init__(self):
		self.last = self.millis()
		MenuOption.__init__(self)
	
	def redraw(self, menu):
		lcd.clear()
		lcd.set_cursor_position(3,1)
		lcd.write("Bye (^_^)/")
		for x in reversed(range(127)):
			backlight.rgb(0, x * 2, 0)
		lcd.clear()
		os.system("halt")
		sys.exit(0)

class Reboot(MenuOption):
	def __init__(self):
		self.last = self.millis()
		MenuOption.__init__(self)
	
	def redraw(self, menu):
		lcd.clear()
		lcd.set_cursor_position(3,1)
		lcd.write("Bye (^_^)/")
		for x in reversed(range(127)):
			backlight.rgb(0, x * 2, 0)
		lcd.clear()
		os.system("reboot")
		sys.exit(0)

class QuitScript(MenuOption):
	def __init__(self):
		self.last = self.millis()
		MenuOption.__init__(self)
	
	def redraw(self, menu):
		lcd.clear()
		lcd.set_cursor_position(3,1)
		lcd.write("Bye (^_^)/")
		for x in reversed(range(127)):
			backlight.rgb(0, x * 2, 0)
		lcd.clear()
		sys.exit(0)