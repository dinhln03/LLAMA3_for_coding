# -*- coding: utf-8 -*-
"""
SHT21 Sensor Plugin.

Return temperature and relative humidity from sensor readings.

Calculate and return absolute humidity and dew point.

Source for calculations:
http://www.vaisala.com/Vaisala%20Documents/Application%20notes/Humidity_Conversion_Formulas_B210973EN-F.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from fcntl import flock, LOCK_EX, LOCK_UN
import math

import collectd
from sht21 import SHT21

sht21 = None
lock_file = None
kock_handle = None


def pws_constants(t):
    """Lookup-table for water vapor saturation pressure constants (A, m, Tn)."""
    if t < -20:
        raise ValueError('Temperature out of range (-20 - 350°C')
    if t < 50:
        return (6.116441, 7.591386, 240.7263)
    if t < 100:
        return (6.004918, 7.337936, 229.3975)
    if t < 150:
        return (5.856548, 7.27731, 225.1033)
    if t < 200:
        return (6.002859, 7.290361, 227.1704)
    return (9.980622, 7.388931, 263.1239)


def pws(t):
    r"""
    Calculate water vapor saturation pressure based on temperature (in hPa).

        P_{WS} = A \cdot 10^{\frac{m \cdot T}{T + T_n}}

    """
    A, m, Tn = pws_constants(t)  # noqa:N806
    power = (m * t) / (t + Tn)
    return A * 10 ** power


def pw(t, rh):
    r"""
    Calculate Pw (in hPa).

        P_W = P_{WS} \cdot RH / 100

    """
    return pws(t) * rh / 100


def td(t, rh):
    r"""
    Calculate the dew point (in °C).

        T_d = \frac{T_n}{\frac{m}{log_{10}\left(\frac{P_w}{A}\right)} - 1}

    """
    A, m, Tn = pws_constants(t)  # noqa:N806
    Pw = pw(t, rh)  # noqa:N806
    return Tn / ((m / math.log(Pw / A, 10)) - 1)


def celsius_to_kelvin(celsius):
    return celsius + 273.15


def ah(t, rh):
    r"""
    Calculate the absolute humidity (in g/m³).

        A = C \cdot P_w / T

    """
    C = 2.16679  # noqa:N806
    Pw = pw(t, rh)  # noqa:N806
    T = celsius_to_kelvin(t)  # noqa:N806
    return C * (Pw * 100) / T


def config(config):
    global lock_file

    for node in config.children:
        key = node.key.lower()
        val = node.values[0]

        if key == 'lockfile':
            lock_file = val
            collectd.info('sht21 user-mode plugin: Using lock file %s' %
                          lock_file)


def init():
    global sht21, lock_file, lock_handle

    if lock_file:
        # Try to open lock file, in case of failure proceed without locking
        try:
            lock_handle = open(lock_file, 'w')
        except IOError as e:
            collectd.error('sht21 plugin: Could not open lock file: %s' % e)
            collectd.error('Proceeding without locking')

    try:
        sht21 = SHT21(1)
        collectd.info('sht21 user-mode plugin initialized')
    except IOError as e:
        collectd.error('sht21 plugin: Could not initialize: %s' % e)
        collectd.unregister_read(read)


def read():
    # Read values
    global sht21, lock_handle
    try:
        if lock_handle:
            flock(lock_handle, LOCK_EX)
        temperature = sht21.read_temperature()
        humidity = sht21.read_humidity()
        if lock_handle:
            flock(lock_handle, LOCK_UN)
    except IOError as e:
        collectd.error('sht21 plugin: Could not read sensor data: %s' % e)
        return

    # Calculate values
    try:
        dewpoint = td(temperature, humidity)
    except ValueError as e:
        collectd.error('sht21 plugin: Could not calculate dew point: %s' % e)
        dewpoint = 0
    absolute_humidity = ah(temperature, humidity)

    # Dispatch values
    v_tmp = collectd.Values(plugin='sht21', type='temperature', type_instance='current')
    v_tmp.dispatch(values=[temperature])
    v_hum = collectd.Values(plugin='sht21', type='humidity', type_instance='relative_humidity')
    v_hum.dispatch(values=[humidity])
    v_abs = collectd.Values(plugin='sht21', type='gauge', type_instance='absolute_humidity')
    v_abs.dispatch(values=[absolute_humidity])
    v_dew = collectd.Values(plugin='sht21', type='temperature', type_instance='dewpoint')
    v_dew.dispatch(values=[dewpoint])


collectd.register_config(config)
collectd.register_init(init)
collectd.register_read(read)
