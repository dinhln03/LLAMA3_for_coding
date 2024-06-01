from Methods import MetaTraderDataConverter, Ichimoku, Ichimoku_plot

import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mpl_finance import candlestick_ohlc
import pandas as pd
import numpy as np

datafile="USDJPY_H1_2014_2018.csv"

df = MetaTraderDataConverter(datafile)
#df = Ichimoku(df)

start = df.index[0]
end = df.index[200]

df = df.iloc[200:400]
df = Ichimoku(df)
df = df.dropna()
print(df)
Ichimoku_plot(df)