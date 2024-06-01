#! /usr/bin/env python 3

import os
import sublime
import sublime_plugin
import random
import re
import sys
import math






__version__ = '0.1.0'
__authors__ = ['Ryan Grannell (@RyanGrannell)']





class BabelCommand (sublime_plugin.WindowCommand):
	"""
	babel loads a random file from your
	currently open folders.
	"""

	def run (self):

		window       = self.window
		open_folders = window.folders()

		# todo
