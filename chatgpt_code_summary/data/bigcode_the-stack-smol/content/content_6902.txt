# -*- coding: utf-8 -*-
"""
Source: https://github.com/awesto/django-shop/blob/12e246b356dbc1bc5bbdc8f056e3cb109c617997/shop/money/__init__.py
"""
from .money_maker import MoneyMaker, AbstractMoney

# The default Money type for this shop
Money = MoneyMaker()
