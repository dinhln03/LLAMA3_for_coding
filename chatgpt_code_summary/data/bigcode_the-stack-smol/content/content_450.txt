#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/2/9 12:09 下午
# @Author: zhoumengjie
# @File  : tabledrawer.py

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

def draw_table(columns_head:[], cell_vals=[]):
    # 设置字体及负数
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 画布
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

    # 数据
    data = [
        [100, 200, 300, -100, 350],
        [-120, 290, -90, 450, 150]
    ]

    # 列与行
    columns = ('一', '二', '三', '四', '五')
    rows = ['A', 'B']

    # 作图参数
    index = np.arange(len(columns)) - 0.1
    bar_width = 0.4

    # 设置颜色
    colors = ['turquoise', 'coral']

    # 柱状图
    bar1 = plt.bar(index, data[0], bar_width, color=colors[0], edgecolor='grey')
    bar2 = plt.bar(index + bar_width, data[1], bar_width, color=colors[1], edgecolor='grey')

    # 设置标题
    ax.set_title('收益情况', fontsize=16, y=1.1, x=0.44)
    ax.set_ylabel('元', fontsize=12, color='black', alpha=0.7, rotation=360)
    ax.set_ylim(-150, 500)

    # 显示数据标签
    # ax.bar_label(bar1, label_type='edge')
    # ax.bar_label(bar2, label_type='edge')

    # x,y刻度不显示
    ax.tick_params(axis=u'both', which=u'both', length=0)
    plt.xticks([])

    table = plt.table(cellText=data, rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns, cellLoc='center', loc='bottom',
                      bbox=[0, -0.4, 1, 0.24])

    cellDict = table.get_celld()
    for i in range(0, len(columns)):
        cellDict[(0, i)].set_height(0.6)
        for j in range(1, len(rows) + 1):
            cellDict[(j, i)].set_height(0.4)

    cellDict[(1, -1)].set_height(0.4)
    cellDict[(2, -1)].set_height(0.4)

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    name = ['', '']
    ax.legend(name, handlelength=0.7, labelspacing=0.6,
              bbox_to_anchor=(-0.1, -0.23), loc='upper left', frameon=False)

    plt.show()


if __name__ == '__main__':
    # draw_table(['A', 'B'], [['中国', '必胜'], ['你好', '谢谢']])
    # print(4800 / 1100 / 1000)
    data = {
        'linux': [1.2, 2.2, 3.1, '中国', 2.0, 1.0, 2.1, 3.5, 4.0, 2.0, ],
        'linuxmi': [5.2, 6.7, 7.9, 8.3, 1.2, 5.7, 6.1, 7.2, 8.3, '-', ],
    }

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(3, 3))

    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=df.values,
             colLabels=df.columns,
             bbox=[0, 0, 1, 1],
             )
    # plt.savefig('xx.png')
    plt.show()
