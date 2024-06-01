""" A class that can provide a date/time in any timeformat.format() format and both
	local and UTC timezones within a ContextVariable.

	Copyright (c) 2004 Colin Stewart (http://www.owlfish.com/)
	All rights reserved.
		
	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions
	are met:
	1. Redistributions of source code must retain the above copyright
	   notice, this list of conditions and the following disclaimer.
	2. Redistributions in binary form must reproduce the above copyright
	   notice, this list of conditions and the following disclaimer in the
	   documentation and/or other materials provided with the distribution.
	3. The name of the author may not be used to endorse or promote products
	   derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
	IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
	OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
	IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
	INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
	NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
	DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
	THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
	THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	
	If you make any bug fixes or feature enhancements please let me know!

"""
import re, time, math, string
import timeformat
from simpletal import simpleTALES

PATHREGEX = re.compile ('^((?:local)|(?:utc))/?(.*)$')

class Date (simpleTALES.ContextVariable):
	""" Wraps a DateTime and provides context paths local and utc.
		These paths in turn can take TimeFormat formats, for example:
			utc/%d-%m-%Y
			
	"""
	def __init__ (self, value = None, defaultFormat = '%a[SHORT], %d %b[SHORT] %Y %H:%M:%S %Z'):
		""" The value should be in the LOCAL timezone.
		"""
		self.ourValue = value
		self.defaultFormat = defaultFormat
		
	def value (self, currentPath=None):
		# Default to local timezone and RFC822 format
		utcTime = 0
		strFrmt = self.defaultFormat
		if (currentPath is not None):
			index, paths = currentPath
			currentPath = '/'.join (paths[index:])
			match = PATHREGEX.match (currentPath)
			if (match is not None):
				type = match.group(1)
				if (type == 'local'):
					utcTime = 0
				else:
					utcTime = 1
				strFrmt = match.group(2)
				if (strFrmt == ""):
					strFrmt = self.defaultFormat
			
		if (self.ourValue is None):
			# Default to the current time!
			timeValue = time.localtime()
		else:
			timeValue = self.ourValue
			
		if (utcTime):
			# Convert to UTC (GMT)
			timeValue = time.gmtime (time.mktime (timeValue))
		value = timeformat.format (strFrmt, timeValue, utctime=utcTime)
		raise simpleTALES.ContextVariable (value)
