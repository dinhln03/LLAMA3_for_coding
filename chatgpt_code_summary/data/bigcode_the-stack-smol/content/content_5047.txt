#!/usr/bin/python

from Adafruit_CharLCDPlate import Adafruit_CharLCDPlate
from subprocess import * 
from time import sleep, strftime
from datetime import datetime
from mpd import *
import threading
import signal
import sys
import os
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import json

PLAY=0
PAUSE=1
STOP=2
VOL=3

LCDon=4

lcd = Adafruit_CharLCDPlate()  # create LCD object
client = MPDClient()           # create MPD client object

lock = threading.Lock()
home=os.path.dirname(os.path.realpath(__file__))

class pimp3clock_HTTPRequesthandler(BaseHTTPRequestHandler):
  def do_GET(self):
    try:
      if '?' in self.path:
        self.path,q = self.path.split('?', 1)
      if self.path.endswith(".js") or self.path.endswith(".css") or self.path.endswith(".png") or self.path.endswith(".gif") or self.path.endswith(".html"):
        f = open(home + "/web/" + self.path)
        self.send_response(200)
        if self.path.endswith(".js"):
          self.send_header('Content-type', 'text/javascript')
        elif self.path.endswith(".css"):
          self.send_header('Content-type', 'text/css')
        elif self.path.endswith(".png"):
          self.send_header('Content-type', 'image/png')
        elif self.path.endswith(".gif"):
          self.send_header('Content-type', 'image/gif')
        elif self.path.endswith(".html"):
          self.send_header('Content-type', 'text/html')
        
        self.end_headers()
        self.wfile.write(f.read())
        f.close()
        return
      elif self.path.endswith(".json"):
        self.send_response(200)
        self.send_header('Content-type',        'text/javascript')
        self.end_headers()
        if self.path.endswith("status.json"):
          lock.acquire()
          song = client.currentsong()
          status = client.status() 
          lock.release()
          self.wfile.write(json.dumps({'song': song, 'status': status}))
          return
        elif self.path.endswith("select.json"):
          lock.acquire()
          status = client.status()
          if status['state'] == "stop":
            client.play()
          elif status['state'] == "play":  
            client.pause(1)
          elif status['state'] == "pause":
            client.pause(0)
          lock.release()
                                                             
          self.wfile.write(json.dumps("OK"))
          return
        elif self.path.endswith("next.json"):
          lock.acquire()
          client.next()
          lock.release()
                                                             
          self.wfile.write(json.dumps("OK"))
          return
        elif self.path.endswith("previous.json"):
          lock.acquire()
          client.previous()
          lock.release()
                                                             
          self.wfile.write(json.dumps("OK"))
          return
        elif self.path.endswith("volume.json"):
          key, value = q.split('=',1)
          if (value < 1):
            value=1
          lock.acquire()          
          client.setvol(value)
          lock.release()   
                                                                   
          self.wfile.write(json.dumps("OK"))
          return
        elif self.path.endswith("update.json"):
          lock.acquire()
          mpd_update()
          lock.release()
          return
        elif self.path.endswith("background.json"):
        	lock.acquire()
        	key, value = q.split('=',1)
        	LCDon=int(value)
        	lcd.backlight(LCDon)
        	lock.release()
        	return
        return
        
      else:
        self.send_response(301)
        self.send_header('Location',	'index.html')
        self.end_headers()
        return
      return
    
    except IOError:
      self.send_error(404,'File Not Found: {0} (Home: {1})'.format(self.path, home))
      
  def do_POST(self):
    try:
      print "POST"
    except:
      pass

def mpd_update():
  # Load Database into current playlist
  client.update()
  client.clear()
  database=client.listall("/")
  for (i) in range(len(database)):
    if 'file' in database[i]:
      client.add(database[i]['file'])
  client.random(1)
  client.shuffle(1)
  client.crossfade(2)
  

def display_lcd(title_a,st_a,vol_a):

  LCDoff=lcd.OFF
  LCDState=LCDoff

  LCDOffDelay=30
  LCDOffCountdown=LCDOffDelay

  lcd.backlight(LCDon)
  lcd.clear()
  lcd.begin(16,1)

  play=[
    0b10000,
    0b11000,
    0b11100,
    0b11110,
    0b11100,
    0b11000,
    0b10000,
    0b00000
    ]
  lcd.createChar(PLAY,play)
  
  pause=[
    0b11011,
    0b11011,
    0b11011,
    0b11011,
    0b11011,
    0b11011,
    0b11011,
    0b11011
    ]
  lcd.createChar(PAUSE,pause)
  
  stop=[
    0b00000,
    0b11111,
    0b10001,
    0b10001,
    0b10001,
    0b10001,
    0b11111,
    0b00000
    ]
  lcd.createChar(STOP,stop)

  
  t=0
  i=0
  fr=1
  oldtitle=""

  while 1:
    lock.acquire()
    vol=[]

    vol.append([0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b00000])
    vol.append([0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b10000,0b10000])
    vol.append([0b00000,0b00000,0b00000,0b00000,0b00000,0b01000,0b11000,0b11000])
    vol.append([0b00000,0b00000,0b00000,0b00000,0b00100,0b01100,0b11100,0b11100])
    vol.append([0b00000,0b00000,0b00000,0b00010,0b00110,0b01110,0b11110,0b11110])
    vol.append([0b00000,0b00000,0b00001,0b00011,0b00111,0b01111,0b11111,0b11111])
    
    volbar=int((vol_a[0]+5)/(100/5))
    lcd.createChar(VOL,vol[volbar])  

    try:
      if (t % 2) == 0:
        lcd.home()
        lcd.write(VOL,True) # Special Characters
        lcd.message(datetime.now().strftime('%d.%b %H:%M:%S'))
      else:
        title=title_a[0]
        if title != oldtitle:
          fr=1
          i=0
        oldtitle=title
        st=st_a[0]
        lcd.clear()
        lcd.write(VOL,True) # Special Characters
        lcd.message(datetime.now().strftime('%d.%b %H %M %S\n'))
        lcd.write(st,True) # Special Characters
        lcd.message('%s' % (title[i:15+i]) )
                
        if ((st == PAUSE) or (st == STOP)):
        	LCDOffCountdown=LCDOffCountdown-1
        else:
        	if (LCDOffCountdown==0):
        		lcd.backlight(LCDon)
        	LCDOffCountdown=LCDOffDelay
        	
        if (LCDOffCountdown < 1):
        	lcd.backlight(LCDoff)
        	LCDOffCountdown=0
        
        if fr==1:
          i=i+1
        else:
          i=i-1
          
        if i>len(title)-15:
          fr=0
        if i==0:
          fr=1
          
    finally:
      lock.release()
      
    t=t+1
    sleep(0.5)
 
def webserver():
  server.serve_forever()

def main_loop():
  i=0;
  title_a=[None]
  st_a=[None]
  vol_a=[None]
  
  title_a[0]=""
  st_a[0]=STOP
  vol_a[0]=0

  display_thread = threading.Thread(target=display_lcd, args=(title_a,st_a,vol_a))
  display_thread.daemon=True  # Causing thread to stop when main process ends.
  display_thread.start()

  webserver_thread = threading.Thread(target=webserver, args=())
  webserver_thread.daemon=True  # Causing thread to stop when main process ends.
  webserver_thread.start()

  client.connect("localhost", 6600)  # connect to localhost:6600

  mpd_update()

  last_button=100;

  while 1:
        lock.acquire()
        status = client.status()
        vol_a[0]=int(status['volume'])
        lock.release()
        if (i % 5) == 0:
          lock.acquire()
          song = client.currentsong()
          lock.release()
          if song == {}:
            title_a[0]=""
          else:
            title_a[0]=song['artist'] + " - " + song['title']
          if status['state'] == "stop":
            st_a[0]=STOP
          elif status['state'] == "play":
            st_a[0]=PLAY
          elif status['state'] == "pause":
            st_a[0]=PAUSE
          
        lock.acquire()
        try:
          button = lcd.buttons()
        finally:
          lock.release()
          
        if ((button & 1) == 1) and (last_button != button): # SELECT
           if status['state'] == "stop":
             lock.acquire()
             client.play()
             lock.release()
           elif status['state'] == "play":
             lock.acquire()
             client.pause(1)
             lock.release()
           elif status['state'] == "pause":
             lock.acquire()
             client.pause(0)
             lock.release()
        elif ((button & 2) == 2) and (last_button != button):  # RIGHT
          client.next()
        elif (button & 4) == 4:  # DOWN
          if int(status['volume']) >1:
            lock.acquire()
            client.setvol(int(status['volume']) - 1)
            lock.release()
        elif (button & 8) == 8:  # UP
          if int(status['volume']) <100:
            lock.acquire()
            client.setvol(int(status['volume']) + 1)
            lock.release()
        elif ((button & 16) == 16) and (last_button != button):  # LEFT
          lock.acquire()
          client.previous()
          lock.release()

        last_button=button

        i=i+1;
	sleep(0.1)


def shutdown():
  client.stop()
  client.close()                     # send the close command
  client.disconnect()                # disconnect from the server
  lcd.clear();
  lcd.stop();
  
def sig_handler(signum = None, frame = None):
  shutdown()
  sys.exit(0)
  
try:
  for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT]:
    signal.signal(sig, sig_handler)
    
  server = HTTPServer(('',80), pimp3clock_HTTPRequesthandler)
  main_loop()
except (KeyboardInterrupt, SystemExit):
  shutdown()

