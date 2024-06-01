# -*- coding: utf-8 -*-
# Copyright (c) 2019, TUSHAR TAJNE and contributors
# For license information, please see license.txt

from __future__ import unicode_literals
import frappe
from frappe.model.document import Document
from frappe import _
class District(Document):
	def validate(self):
		name = str(self.district.capitalize())
		self.name = _(name)
	pass
