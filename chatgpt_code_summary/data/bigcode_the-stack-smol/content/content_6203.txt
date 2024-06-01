#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by TaoYuan on 2018/2/28 0028. 
# @Link    : http://blog.csdn.net/lftaoyuan  
# Github   : https://github.com/seeways
import asyncio
import orm
from models import User, Blog, Comment


@asyncio.coroutine
def test(loop):
    yield from orm.create_pool(loop=loop, user='root', password='123456', database='awesome')
    # taoyuan 123456
    u = User(name='TaoYuan', email='taoyuan', passwd='396d447288c288f0ff7ba1fc608600d7e233646d', image='about:blank')
    yield from u.save()


loop = asyncio.get_event_loop()
loop.run_until_complete(test(loop))
loop.close()
