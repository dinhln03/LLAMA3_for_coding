#!/usr/bin/env python3

import RPi.GPIO as GPIO
import time
import threading
import logging
import pandas as pd
import numpy as np
from tzlocal import get_localzone
from flask import Flask, render_template, url_for, request

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
GPIO.setmode(GPIO.BCM)
logger = logging.getLogger(__name__)

from rpiweather import temphumid
from rpiweather import temppressure
from rpiweather import data
from rpiweather import outside_weather
from rpiweather import dust

temppressure.start_recording()
temphumid.start_recording()
outside_weather.start_recording()
dust.start_recording()

app = Flask("rpiweather")


def format_timestamps(series):
    local_tz = get_localzone()
    return list(
        str(dt.tz_localize("UTC").tz_convert(local_tz)) for dt in series
    )


@app.route("/")
def index():
    lookbehind = int(request.args.get('lookbehind', 24))

    bigarray = data.get_recent_datapoints(lookbehind)
    logger.info("Total datapoint count: %d" % len(bigarray))
    df = pd.DataFrame(bigarray, columns=['time', 'type', 'value'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    agg_interval = "15T" if lookbehind < 168 else "1H" if lookbehind < 5040 else "1D"
    df2 = df.pivot(columns='type', values='value').resample(agg_interval).mean()

    temp_df = df2['temperature'].dropna()
    temp_values = {
        'x': format_timestamps(temp_df.index),
        'y': list(temp_df),
        'name': 'Temperature',
        'type': 'line',
        'line': {
            'color': 'rgb(244, 66, 98)'
        }
    }

    outside_temp_df = df2['outside_temperature'].dropna()
    ot_values = {
        'x': format_timestamps(outside_temp_df.index),
        'y': list(outside_temp_df),
        'name': 'Temperature Outside',
        'type': 'line',
        'line': {
            'color': 'rgb(244, 66, 98)',
            'dash': 'longdash'
        }
    }

    pres_df = df2['pressure'].dropna()
    pressure_values = {
        'x': format_timestamps(pres_df.index),
        'y': list(pres_df),
        'name': 'Pressure',
        'type': 'line',
        'yaxis': 'y2',
        'line': {
            'dash': 'dot',
            'color': 'rgb(151,138,155)'
        }
    }
    hum_df = df2['humidity'].dropna()
    humidity_values = {
        'x': format_timestamps(hum_df.index),
        'y': list(hum_df),
        'name': 'Humidity',
        'type': 'scatter',
        'fill': 'tozeroy',
        'yaxis': 'y3',
        'marker': {
            'color': 'rgb(66,131,244)'
        }
    }
    dust_df = df2['dust'].dropna()
    dust_values = {
        'x': format_timestamps(dust_df.index),
        'y': list(dust_df),
        'name': 'Dust level',
        'type': 'line',
        'yaxis': 'y4',
        'line': {
            'dash': 'dot',
            'color': 'rgb(224, 205, 31)'
        }
    }

    chart_data = [
        temp_values, pressure_values, humidity_values, ot_values, dust_values
    ]

    #import pdb; pdb.set_trace()
    lookbehind_options = [(24, "1d"),
               (24*7, "1w"),
               (24*7*30, "30d")]
    return render_template("index.html",
                           weather_data=chart_data,
                           lookbehind_options=lookbehind_options,
                           lookbehind=lookbehind)


def make_agg_df(rec):
    df = pd.DataFrame.from_records(rec, index="time")
    df.index = pd.to_datetime(df.index, unit="s")
    return df.resample("T").mean()


def magic():
    df_tp = make_agg_df(temppressure.get_records())
    df_th = make_agg_df(temphumid.get_records())
    df_th = df_th.rename(columns={'temp': 'bad_temp'})
    total_view = pd.concat([df_tp, df_th], axis=1)
    return total_view


#import IPython
# IPython.embed()
if False:
    bigarray = data.get_recent_datapoints()
    df = pd.DataFrame(bigarray, columns=['time', 'type', 'value'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df2 = df.pivot(columns='type', values='value').resample("5T").mean()

    temp_values = list(zip(
        (dt.timestamp() for dt in df2.index),
        df2['temperature']
    ))
    pressure_values = list(zip(
        (dt.timestamp() for dt in df2.index),
        df2['pressure']
    ))
    humidity_values = list(zip(
        (dt.timestamp() for dt in df2.index),
        df2['humidity']
    ))
