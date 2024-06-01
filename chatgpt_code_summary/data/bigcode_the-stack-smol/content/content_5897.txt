# coding: utf-8

import codecs
import re
import json
from budget2013_common import *


class Budget2013_37_SubTable1Item(object):
	def __init__(self):
		self._no = None
		self._purpose = None
		self._principal = None
		self._value = None
		self._regress = None
		self._check = None
		self._other = []
	
	@property
	def no(self):
		return self._no
	@no.setter
	def no(self, value):
		self._no = value
	
	@property
	def purpose(self):
		return self._purpose
	@purpose.setter
	def purpose(self, value):
		self._purpose = value
	
	@property
	def principal(self):
		return self._principal
	@principal.setter
	def principal(self, value):
		self._principal = value
	
	@property
	def value(self):
		return self._value
	@value.setter
	def value(self, value):
		self._value = value
	
	@property
	def regress(self):
		return self._regress
	@regress.setter
	def regress(self, value):
		self._regress = value
	
	@property
	def check(self):
		return self._check
	@check.setter
	def check(self, value):
		self._check = value
	
	@property
	def other(self):
		return self._other
	@other.setter
	def other(self, value):
		self._other = value

class JsonEncoder_Budget2013_37_SubTable1Item(json.JSONEncoder):
	def default(self, o):
		return {
			"no": o.no,
			"purpose": o.purpose,
			"principal": o.principal,
			"value": o.value,
			"regress": o.regress,
			"check": o.check,
			"other": o.other
		}

class Budget2013_37_SubTable1(object):
	def __init__(self):
		self._caption = None
		self._headers = []
		self._items = []
		self._notes = []
	
	@property
	def caption(self):
		return self._caption
	@caption.setter
	def caption(self, value):
		self._caption = value
	
	@property
	def headers(self):
		return self._headers
	@headers.setter
	def headers(self, value):
		self._headers = value
	
	@property
	def items(self):
		return self._items
	@items.setter
	def items(self, value):
		self._items = value

	@property
	def notes(self):
		return self._notes
	@notes.setter
	def notes(self, value):
		self._notes = value

class JsonEncoder_Budget2013_37_SubTable1(json.JSONEncoder):
	def default(self, o):
		item_encoder = JsonEncoder_Budget2013_37_SubTable1Item()
		return {
			"caption": o.caption,
			"headers": o.headers,
			"items": [item_encoder.default(item) for item in o.items],
			"notes": o.notes
		}

class Budget2013_37_SubTable2(object):
	def __init__(self):
		self._caption = None
		self._headers = []
		self._items = []
	
	@property
	def caption(self):
		return self._caption
	@caption.setter
	def caption(self, value):
		self._caption = value
	
	@property
	def headers(self):
		return self._headers
	@headers.setter
	def headers(self, value):
		self._headers = value
	
	@property
	def items(self):
		return self._items
	@items.setter
	def items(self, value):
		self._items = value


class JsonEncoder_Budget2013_37_SubTable2Item(json.JSONEncoder):
	def default(self, o):
		return {
			"name": o["name"],
			"value": o["value"]
		}

class JsonEncoder_Budget2013_37_SubTable2(json.JSONEncoder):
	def default(self, o):
		item_encoder = JsonEncoder_Budget2013_37_SubTable2Item()
		return {
			"caption": o.caption,
			"headers": o.headers,
			"items": [item_encoder.default(item) for item in o.items]
		}

class Budget2013_37(object):
	def __init__(self):
		self._caption = None
		self._subtable1 = Budget2013_37_SubTable1()
		self._subtable2 = Budget2013_37_SubTable2()
	
	@property
	def caption(self):
		return self._caption
	@caption.setter
	def caption(self, value):
		self._caption = value
	
	@property
	def subtable1(self):
		return self._subtable1
	
	@property
	def subtable2(self):
		return self._subtable2

class JsonEncoder_Budget2013_37(json.JSONEncoder):
	def default(self, o):
		subtable1_encoder = JsonEncoder_Budget2013_37_SubTable1()
		subtable2_encoder = JsonEncoder_Budget2013_37_SubTable2()
		return {
			"caption": o.caption,
			"subtable1": subtable1_encoder.default(o.subtable1),
			"subtable2": subtable2_encoder.default(o.subtable2)
		}

def check_document(document):
	total_value = 0.0
	for item in document.subtable1.items[:-1]:
		total_value += item.value
	if total_value != document.subtable1.items[-1].value:
		print total_value, document.subtable1.items[-1].value
		raise Exception(u"Сумма не сходится.")

def get_document(input_file_name):
	with codecs.open(input_file_name, "r", encoding = "utf-8-sig") as input_file:
		input_data = input_file.readlines()
		
		document = Budget2013_37()
		
		line_index = 0
		
		# caption
		caption_lines = []
		while line_index < len(input_data):
			caption_line = input_data[line_index].strip()
			line_index += 1
			if not caption_line:
				break
			caption_lines.append(caption_line)
		document.caption = join_lines(caption_lines)

		# subtable1 caption
		caption_lines = []
		while line_index < len(input_data):
			caption_line = input_data[line_index].strip()
			line_index += 1
			if not caption_line:
				break
			caption_lines.append(caption_line)
		document.subtable1.caption = join_lines(caption_lines)
		
		# subtable1 headers
		headers = input_data[line_index].strip()
		line_index += 2
		document.subtable1.headers = headers.split(";")
		
		# subtable1 data
		while not input_data[line_index].strip().startswith(u"ИТОГО"):
			item = Budget2013_37_SubTable1Item()
			
			# no + purpose
			purpose_lines = []
			while line_index < len(input_data):
				purpose_line = input_data[line_index].strip()
				line_index += 1
				if not purpose_line:
					break
				purpose_lines.append(purpose_line)
			purpose = join_lines(purpose_lines)
			m = re.compile(u"^(\\d+) (.*)").match(purpose)
			item.no = int(m.group(1))
			item.purpose = m.group(2)
			
			# principal
			principal_lines = []
			while line_index < len(input_data):
				principal_line = input_data[line_index].strip()
				line_index += 1
				if not principal_line:
					break
				principal_lines.append(principal_line)
			item.principal = join_lines(principal_lines)
			
			# value
			item.value = float(input_data[line_index].strip().replace(",", ".").replace(" ", ""))
			line_index += 2
			
			# regress
			s = input_data[line_index].strip()
			if s == u"Нет":
				item.regress = False
			elif s == u"Есть":
				item.regress = True
			else:
				print s
				raise Exception(u"Unknown regress: " + s)
			line_index += 2
			
			# check
			s = input_data[line_index].strip()
			if s == u"Нет":
				item.check = False
			elif s == u"Есть":
				item.check = True
			else:
				print s
				raise Exception(u"Unknown check: " + s)
			line_index += 2
			
			# other
			other_lines = []
			while line_index < len(input_data):
				other_line = input_data[line_index].strip()
				line_index += 1
				if not other_line:
					break
				if re.compile("^\\d+\\. ").match(other_line):
					if other_lines:
						o = join_lines(other_lines)
						item.other.append(o)
						other_lines = []
				other_lines.append(other_line)
			if other_lines:
				o = join_lines(other_lines)
				item.other.append(o)
				other_lines = []
			
			document.subtable1.items.append(item)
		
		# ИТОГО
		s = input_data[line_index].strip()
		m = re.compile(u"^(ИТОГО)\\*? (.*)").match(s)
		item = Budget2013_37_SubTable1Item()
		item.purpose = m.group(1)
		item.value = float(m.group(2).replace(",", ".").replace(" ", ""))
		document.subtable1.items.append(item)
		line_index += 2
		
		# notes
		notes_lines = []
		while line_index < len(input_data):
			notes_line = input_data[line_index].rstrip()
			line_index += 1
			if not notes_line:
				break
			m = re.compile("^\\*? (.*)").match(notes_line)
			if m:
				if notes_lines:
					note = join_lines(notes_lines)
					document.subtable1.notes.append(note)
					notes_lines = []
				notes_lines.append(m.group(1))
			else:
				notes_lines.append(notes_line.lstrip())
		if notes_lines:
			note = join_lines(notes_lines)
			document.subtable1.notes.append(note)
			notes_lines = []
		
		line_index += 1
		
		# subtable2 caption
		caption_lines = []
		while line_index < len(input_data):
			caption_line = input_data[line_index].strip()
			line_index += 1
			if not caption_line:
				break
			caption_lines.append(caption_line)
		document.subtable2.caption = join_lines(caption_lines)

		# subtable2 headers
		headers = input_data[line_index].strip()
		line_index += 1
		document.subtable2.headers = headers.split(";")

		#subtable2 data
		while line_index < len(input_data):
			data_line = input_data[line_index].strip()
			line_index += 1
			if not data_line:
				break
			m = re.compile("([\\d ,]+)$").search(data_line)
			value = float(m.group(1).replace(",", ".").replace(" ", ""))
			name = data_line[:len(data_line) - len(m.group(1)) - 1].strip()
			item = {"name": name, "value": value}
			document.subtable2.items.append(item)
		
		check_document(document)
		
		return document

def do_write_text_document(output_file, document):
	output_file.write(document.caption + "\r\n\r\n")
	
	output_file.write(document.subtable1.caption + "\r\n\r\n")
	output_file.write(u" ".join(document.subtable1.headers) + "\r\n\r\n")
	for item in document.subtable1.items[:-1]:
		output_file.write(unicode(item.no) + " " + item.purpose + " " + 
			item.principal + " " + unicode(item.value) + " " + 
			unicode(item.regress) + " " + unicode(item.check) + "\r\n")
		if item.other:
			for o in item.other:
				output_file.write(o + "\r\n");
		output_file.write("\r\n")
	output_file.write(document.subtable1.items[-1].purpose + " " + unicode(document.subtable1.items[-1].value) + "\r\n\r\n")
	for note in document.subtable1.notes:
		output_file.write(note + "\r\n")
	output_file.write("\r\n")

	output_file.write(document.subtable2.caption + "\r\n\r\n")
	output_file.write(u" ".join(document.subtable2.headers) + "\r\n\r\n")
	for item in document.subtable2.items:
		output_file.write(item["name"] + " " + unicode(item["value"]) + "\r\n")


if __name__ == "__main__":
	parser = get_default_argument_parser()
	args = parser.parse_args()
	
	input_file_name = args.input_file_name
	output_pickle_file_name = args.output_pickle_file_name
	output_text_file_name = args.output_text_file_name
	output_json_file_name = args.output_json_file_name
	output_json_pretty_file_name = args.output_json_pretty_file_name
	
	if (not output_pickle_file_name) and (not output_text_file_name) and (not output_json_file_name) and (not output_json_pretty_file_name):
		raise Exception("No output file specified")
	
	document = get_document(input_file_name)
	if output_pickle_file_name:
		write_pickle_document(document, output_pickle_file_name)
	if output_text_file_name:
		write_text_document(document, output_text_file_name, do_write_text_document)
	if output_json_file_name:
		write_json_document(document, output_json_file_name, JsonEncoder_Budget2013_37)
	if output_json_pretty_file_name:
		write_json_pretty_document(document, output_json_pretty_file_name, JsonEncoder_Budget2013_37)
