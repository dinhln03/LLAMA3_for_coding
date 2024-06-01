# _*_ coding: utf-8 _*_
__author__ = 'Di Meng'
__date__ = '1/3/2018 10:16 PM'

# _*_ coding: utf-8 _*_
__author__ = 'Di Meng'
__date__ = '1/3/2018 9:26 PM'

from tutorial.feature_functions import *
import pandas as pd
import plotly as py
import json

from plotly import tools
import plotly.graph_objs as go

#loading our data
df = pd.read_csv('EURUSD_hours.csv')
df.columns = ['date','open','high','low','close','volume']
df.date = pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)
df = df[['open','high','low','close','volume']]
df.drop_duplicates(keep=False)
df = df.iloc[:500]

#moving average
ma = df.close.rolling(center=False, window=30).mean()


# detrended = detrend(df, method='difference')

# f = fourier(df, [10, 15],method='difference')

#HA
# HAresults = candles(df, [1])
# HA = HAresults.candles[1]

#wad
results = wadl(df, [15])
line = results.wadl[15]
print(line['close'])

# draw grarphs
trace = go.Ohlc(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name='Currency Quote')
trace1 = go.Scatter(x=df.index, y=ma)

trace2 = go.Scatter(x=df.index, y=(line.close.to_json()))

# linear detrand plot
# trace2 = go.Scatter(x=df.index, y=detrended)

# difference detrand plot
# trace2 = go.Scatter(x=df.index, y=detrended)




data = [trace, trace1, trace2]
fig = tools.make_subplots(rows=2,cols=1,shared_xaxes=True)
fig.append_trace(trace,1,1)
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,2,1)

py.offline.plot(fig, filename="test.html")