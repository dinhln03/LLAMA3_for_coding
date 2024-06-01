#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/25 0025 上午 10:14
# @Author  : Exchris Tsai
# @Site    : 
# @File    : example52.py
# @Software: PyCharm

"""
题目：学习使用按位或 | 。
程序分析：0|0=0; 0|1=1; 1|0=1; 1|1=1
"""

__author__ = 'Exchris Tsai'

if __name__ == '__main__':
    a = 0o77
    b = a | 3
    print('a | b is %d' %b)
    b |= 7
    print('a | b is %d' %b)