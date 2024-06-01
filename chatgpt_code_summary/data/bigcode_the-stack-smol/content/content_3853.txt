"""
Filename: RobotsParser.py
Author: Maxwell Goldberg
Last modified: 06.09.17
Description: Helper class for parsing individual robots.txt records.
"""

# CONSTANTS
from constants import RECORD_MAX_LEN
# PYTHON BUILTINS
import re, unicodedata, logging

def test_ctrl_chars(s):
	return len(s) != len("".join(ch for ch in s if unicodedata.category(ch)[0]!="C"))

class RobotsParser:
	valid_fields = [u'user-agent', u'allow', u'disallow']

	def __init__(self, record=None):
		if record is None:
			raise TypeError('Parameter record must not be NoneType')
		if not isinstance(record, unicode):
			raise TypeError('Parameter record must be a Unicode string')
		if len(record) > RECORD_MAX_LEN:
			raise ValueError('Parameter record exceeds maximum record num characters')
		self.record = record


	def parse_field(self, field):
		field = field.strip().lower()
		if field not in RobotsParser.valid_fields:
			raise ValueError('Record contains invalid field')
		return field


	def parse_path(self, path):
		path = path.strip()
		if test_ctrl_chars(path):
			raise ValueError('Record path contains control characters')
		# Get path length prior to parsing
		self.init_path_len = len(path)
		path = re.escape(path)
		path = path.replace('\\*', '.*').replace('\\$', '$')
		return path

	def parse(self):
		# Attempt to separate a record by a colon delimiter.
		record_list = self.record.split('#')[0]
		record_list = record_list.split(':', 1)

		if len(record_list) <= 1:
			raise ValueError('Record must contain a delimiter')
		if len(record_list) > 2:
			raise ValueError('Record contains too many delimited fields')
		# Parse the field
		self.field = self.parse_field(record_list[0])
		# Parse the path
		self.path = self.parse_path(record_list[1])