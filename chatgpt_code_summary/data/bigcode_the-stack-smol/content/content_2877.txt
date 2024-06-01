#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cplotting as cplot

S={2+2j, 3+2j, 1.75+1j, 2+1j, 2.25+1j, 2.5+1j, 2.75+1j, 3+1j, 3.25+1j}

cplot.plot({1+2j+z for z in S},4)
cplot.show()
