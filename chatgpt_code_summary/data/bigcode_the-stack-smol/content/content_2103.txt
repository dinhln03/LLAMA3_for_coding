# Copyright (C) 2019 Google Inc.
# Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

"""Provides an HTML cleaner function with sqalchemy compatible API"""
import re

import HTMLParser

import bleach


# Set up custom tags/attributes for bleach
BLEACH_TAGS = [
    'caption', 'strong', 'em', 'b', 'i', 'p', 'code', 'pre', 'tt', 'samp',
    'kbd', 'var', 'sub', 'sup', 'dfn', 'cite', 'big', 'small', 'address',
    'hr', 'br', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul',
    'ol', 'li', 'dl', 'dt', 'dd', 'abbr', 'acronym', 'a', 'img',
    'blockquote', 'del', 'ins', 'table', 'tbody', 'tr', 'td', 'th',
] + bleach.ALLOWED_TAGS

BLEACH_ATTRS = {}

ATTRS = [
    'href', 'src', 'width', 'height', 'alt', 'cite', 'datetime',
    'title', 'class', 'name', 'xml:lang', 'abbr'
]

BUGGY_STRINGS_PATTERN = "&.{2,3};"

for tag in BLEACH_TAGS:
  BLEACH_ATTRS[tag] = ATTRS


CLEANER = bleach.sanitizer.Cleaner(
    tags=BLEACH_TAGS, attributes=BLEACH_ATTRS, strip=True
)

PARSER = HTMLParser.HTMLParser()


def cleaner(dummy, value, *_):
  """Cleans out unsafe HTML tags.

  Uses bleach and unescape until it reaches a fix point.

  Args:
    dummy: unused, sqalchemy will pass in the model class
    value: html (string) to be cleaned
  Returns:
    Html (string) without unsafe tags.
  """
  if value is None:
    # No point in sanitizing None values
    return value

  if not isinstance(value, basestring):
    # No point in sanitizing non-strings
    return value

  value = unicode(value)

  buggy_strings = re.finditer(BUGGY_STRINGS_PATTERN, PARSER.unescape(value))

  while True:
    lastvalue = value
    value = PARSER.unescape(CLEANER.clean(value))
    if value == lastvalue:
      break

  # for some reason clean() function converts strings like "&*!;" to "&*;;".
  # if we have such string we are replacing new incorrect values to old ones
  if buggy_strings:
    backup_value = value
    updated_buggy_strings = re.finditer(BUGGY_STRINGS_PATTERN, value)
    for match in updated_buggy_strings:
      try:
        old_value = buggy_strings.next().group()
        start, finish = match.span()
        value = value[:start] + old_value + value[finish:]
      except StopIteration:
        # If we have different number of string after clean function
        # we should skip replacing
        return backup_value

  return value
