#!/usr/bin/env python
# encoding: utf-8

"""
@author: zhanghe
@software: PyCharm
@file: type_bank.py
@time: 2019-08-17 18:23
"""

from __future__ import unicode_literals

from flask_babel import lazy_gettext as _

from app_common.maps.default import DEFAULT_SEARCH_CHOICES_INT, DEFAULT_SELECT_CHOICES_INT

# 银行类型（1:基本账户,2:一般账户）
TYPE_BANK_BASIC = 1
TYPE_BANK_GENERAL = 2

TYPE_BANK_DICT = {
    TYPE_BANK_BASIC: _('Basic Account'),  # 基本账户（对公）
    TYPE_BANK_GENERAL: _('General Account'),  # 一般账户（对公）
}

TYPE_BANK_SELECT_CHOICES = DEFAULT_SELECT_CHOICES_INT + TYPE_BANK_DICT.items()  # 选择
TYPE_BANK_SEARCH_CHOICES = DEFAULT_SEARCH_CHOICES_INT + TYPE_BANK_DICT.items()  # 搜索
