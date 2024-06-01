#!/usr/bin/python3

import gameinstance
from colour import Colour
from constants import Constants
from vec2 import vec2
from misc import Fade
import sdl2
import hud

class Menu(gameinstance.GameInstance):
	"""Game menu representation."""

	# Variables to control menu background.
	backgrounds = []
	current_bg  = 0
	bg_offset_at_start = None #250
	bg_offset   = None #bg_offset_at_start
	bg_dimness_peak    = None #0x88
	bg_dimness_current = None #0xff
	bg_diminish_rate   = None #0.33

	def __init__(self, renderer, highscores=None):
		# Background stuff.
		Menu.bg_offset_at_start = 250
		Menu.bg_offset = Menu.bg_offset_at_start
		Menu.bg_dimness_peak = 0x88
		Menu.bg_dimness_current = 0xff
		Menu.bg_diminish_rate = 0.33

		self.choice = None
		self.is_open = True

		self.title = hud.Text('pyNoid', renderer, Constants.TITLE_FONT_SIZE)
		self.title.position = vec2(50, 50)

		self.credits = hud.Text('Kacper Tonia 2017/18', renderer, Constants.TINY_FONT_SIZE)
		self.credits.position = self.title.position + vec2(self.title.size[0]//2, self.title.size[1])

		grey = Colour.greyscale(0.75)

		sub1 = hud.Button.buildClickableText('New Game', renderer,
			Colour.White, grey, grey, Constants.MENU_FONT_SIZE
		)
		sub2 = hud.Button.buildClickableText('Exit', renderer,
			Colour.White, grey, grey, Constants.MENU_FONT_SIZE
		)
		self.menu = hud.VerticalContainer([sub1, sub2], Constants.WINDOW_SIZE.y//2)

		if highscores:
			leaderboard = []
			player_name_length = max([len(x[0]) for x in highscores])
			score_length = max([len(str(x[1])) for x in highscores])
			s_format = '{:>%d} {}{}' % player_name_length
			for idx,item in enumerate(highscores):
				leaderboard.append( hud.Text(s_format.format(
					item[0], item[1], ' '*(score_length-len(str(item[1])))),
					renderer, Constants.FONT_SIZE_1,
					Colour.greyscale((5-idx) / 5.0 ))
				)
			self.render_content = hud.VerticalContainer(leaderboard, Constants.WINDOW_SIZE.y*3//4)
		else:
			self.render_content = []

	def update(self):
		"""Update game state."""
		Menu.bg_offset += 2
		if Menu.bg_offset > 0.85 * Constants.WINDOW_SIZE.x:
			Menu.bg_dimness_current += Menu.bg_diminish_rate
			if Menu.bg_dimness_current >= 0xff:
				Menu.current_bg = (Menu.current_bg + 1) % len(Menu.backgrounds)
				Menu.bg_offset = Menu.bg_offset_at_start
				Menu.bg_dimness_current = 0xff
		elif Menu.bg_dimness_current > Menu.bg_dimness_peak:
			Menu.bg_dimness_current -= Menu.bg_diminish_rate

	def handleEvent(self, e):
		"""Process relevant events."""
		for i in self.menu.elem:
			i.handleEvent(e)

		if self.menu.elem[0].isPressed():
			self.fading = True
		elif self.menu.elem[1].isPressed():
			self.is_open = False
	
	def render(self, renderer):
		"""Render scene."""
		rect = (Constants.WINDOW_SIZE.x - Menu.bg_offset, 0, *Constants.WINDOW_SIZE)
		renderer.copy(Menu.backgrounds[Menu.current_bg], None, rect)
		renderer.fill((0, 0, Constants.WINDOW_SIZE.x, Constants.WINDOW_SIZE.y), (0, 0, 0, Menu.bg_dimness_current))

		self.title.render(renderer)
		self.credits.render(renderer)
		self.menu.render(renderer)
		if self.render_content:
			self.render_content.render(renderer)
		if self.fading:
			self.fader.draw(renderer)
			if self.fader.finished():
				self.fading = False
				self.fader.reset()
				self.choice = 0

	def isOpen(self):
		"""Returns False if GameInstance should be no longer active."""
		return self.is_open

	def typeOf(self):
		return 'Menu'