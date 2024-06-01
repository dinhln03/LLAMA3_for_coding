#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 生成词云
'''
Reference:
        https://amueller.github.io/word_cloud/
    	https://github.com/amueller/word_cloud
'''

from wordcloud import WordCloud
import matplotlib.pyplot as plt


filename = "***.txt"	# 文本
with open(filename) as f:
	mytext = f.read()

# print(mytext)

wordcloud = WordCloud().generate(mytext)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")  # 隐藏坐标
plt.show()

