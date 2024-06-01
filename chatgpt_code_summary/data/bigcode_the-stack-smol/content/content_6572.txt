import requests
import re
import random
import time

from bs4 import BeautifulSoup
import os

import lpmysql
import json

def getindex():
    url = 'http://freeget.co'
    headers = {'User-Agent': "Mozilla/5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13B143 Safari/601.1",
                "contentType":"text/html;charset=utf-8",
                # Accept:*/*
                # Accept-Encoding:gzip, deflate
                # Accept-Language:zh-CN,zh;q=0.8
                # Connection:keep-alive

    }
    html = requests.get(url,headers=headers) ##这儿更改了一下（是不是发现  self 没见了？）
    print(html.content)
    print(dir(html))
    print(html.headers)

def getbtn():

    url = 'http://freeget.co/video/extraction'
    headers = {'User-Agent': "Mozilla/5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13B143 Safari/601.1",
                "contentType":"text/html;charset=utf-8",
                # "X - CSRFToken":"1504169114##45f7200f8dba99432cc422ed552b3bbf3baff85b",
               "X - Requested - With": "XMLHttpRequest",
               # X - CSRFToken: 1504164180  ##fdbd5ae5ec0c76632937754c20e90c582f2f7c28
               # X - Requested - With: XMLHttpRequest
                # Accept:*/*
                # Accept-Encoding:gzip, deflate
                # Accept-Language:zh-CN,zh;q=0.8
                # Connection:keep-alive

    }
    payload = {"url":"1111111111111111111111111111111111111111111111","_csrf" : "1504169114##45f7200f8dba99432cc422ed552b3bbf3baff85b"}

    html = requests.post(url,headers=headers,data=payload) ##这儿更改了一下（是不是发现  self 没见了？）
    print(html.content)
    print(dir(html))
    print(html.headers)
# getindex()
getbtn()

# http://www.cnblogs.com/xwang/p/3757711.html
# pythonrequests 设置CSRF
# http://blog.csdn.net/u011061889/article/details/72904821









