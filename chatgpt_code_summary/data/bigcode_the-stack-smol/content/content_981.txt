#! /usr/bin/env python3
# -*- coding: utf8 -*-

# Virtual dancers that consumes real GigglePixel packets
#
# To use, start this up and then bring up a server broadcasting GigglePixel.
# When this receives a palette packet, the dancing pair (whose humble wearables
# are only capable of displaying one color at a time apiece) will light up
# to match the first two elements of the packet received. When an ID packet
# is received, they will shout their love of the sender.

PORT = 7016

import socket
import sys
from time import time
from x256 import x256

from udp import *

WHITE = '\033[0m'
RGB1 = None
RGB2 = None
banner = "Yay"
note = u'♪'
face = u'(・o･)'

# Print without newline
def p(s):
  sys.stdout.write(s)

# Return a two-element array showing current arm position, and toggle it for next time
arm_phase = False
def arms():
  global arm_phase
  arm_phase = not arm_phase
  if arm_phase:
    return u'┏┛'
  else:
    return u'┗┓'

# Take an RGB value and return an ANSI escape sequence to show it in the terminal
def color(rgb):
  if rgb is None:
    return ""
  ix = x256.from_rgb(*rgb)
  return "\033[38;5;%dm" % ix

# Draw the dancers
def draw():
  l, r = arms()
  p (color(RGB1) + l + face + r + WHITE + ' ' + note + ' ')
  l, r = arms()
  p (color(RGB2) + l + face + r + WHITE + "  -" + banner + "!")
  p ("\n\033[1A")  # Keep drawing over and over on the same line


def handle_packet(gp):
  global banner
  global RGB1
  global RGB2

  if gp is None: return
  if gp.packet_type == "PALETTE":
    entries = gp.payload["entries"]
    if len(entries) < 1:
      return
    elif len(entries) == 1:
      entries.extend(entries)
    RGB1 = (entries[0]["red"], entries[0]["green"], entries[0]["blue"])
    RGB2 = (entries[1]["red"], entries[1]["green"], entries[1]["blue"])
  elif gp.packet_type == "ID":
    banner = "We love " + gp.payload["name"]

next_dance = time()
listener = GigglePixelListener()
try:
  while True:
    draw()
    now = time()
    time_left = next_dance - now
    gp = None
    if time_left > 0:
      gp = listener.get_packet(time_left)
      handle_packet(gp)
    if gp is None:
      next_dance = time() + 1
      arms()  # Toggle arm positions

except KeyboardInterrupt:
  print (WHITE)
