#!/usr/bin/env python
# -*- coding: utf-8 -*-
import urllib
from lxml import html
import requests
page = requests.get('http://stmary-338.com/')
tree = html.fromstring(page.content)
info = tree.xpath('//*[@id="panel-w5840cbe2b571d-0-1-0"]/div/div/h6[1]')
for i in info:
	    print "ST MARY", i.encode(page.encoding)
