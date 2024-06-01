# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:52:03 2020

@author: DELL
"""


import pandas as pd

data = pd.read_csv('http://187.191.75.115/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip', encoding = 'ANSI')


res = data[data['ENTIDAD_RES'] == 31]

res.to_csv('data_yuc_actualizado.csv', index = False)