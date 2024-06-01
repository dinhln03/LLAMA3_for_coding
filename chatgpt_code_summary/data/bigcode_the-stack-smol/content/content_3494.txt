#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : Airbnb                                                             #
# Version : 0.1.0                                                              #
# File    : split_names.py                                                     #
# Python  : 3.8.0                                                              #
# ---------------------------------------------------------------------------- #
# Author : John James                                                          #
# Company: DecisionScients                                                     #
# Email  : jjames@decisionscients.com                                          #
# ---------------------------------------------------------------------------- #
# Created      : Tuesday, 7th January 2020 10:22:44 am                         #
# Last Modified: Tuesday, 7th January 2020 10:22:44 am                         #
# Modified By  : John James (jjames@decisionscients.com>)                      #
# ---------------------------------------------------------------------------- #
# License: BSD                                                                 #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
#%%
import os
directory = "./data/raw/"
filenames = os.listdir(directory)
for filename in filenames:
    name = filename.split(".")[0]
    print(name)


# %%
