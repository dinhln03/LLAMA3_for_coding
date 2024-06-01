from gpiozero import CPUTemperature
from tabulate import tabulate
from math import floor
import numpy as np
import termplotlib as tpl
import time
import shutil

def roundNum(num, digits):
    return floor(num * 10 ** digits) / (10 ** digits)

def CtoF(temp):
    fahrenheit = (temp + 1.8) + 32
    rounded = roundNum(fahrenheit, 3)
    return str(rounded)

cpu = CPUTemperature()
colors = {
    'HEADER': '\033[95m',
    'OKBLUE': '\033[94m',
    'OKCYAN': '\033[96m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
}

times = [0]
temps = [cpu.temperature]


while True:
    tickRate = 2 #takes data every {tickRate} seconds
    minutes = 5
    numPoints = int(60 / tickRate * minutes)
    width, height = shutil.get_terminal_size()

    if len(temps) > numPoints:
        temps = temps[-numPoints:]
        times = times[-numPoints:]

    temps.append(cpu.temperature)
    times.append(times[-1] + tickRate)

    averageTemp = roundNum(np.average(temps), 3)

    cpuTempColor = ''
    if cpu.temperature < 50:
        cpuTempColor = colors['OKBLUE']
    elif cpu.temperature < 65:
        cpuTempColor = colors['OKCYAN']
    elif cpu.temperature < 80:
        cpuTempColor = colors['OKGREEN']
    else:
        cpuTempColor = colors['FAIL'] + colors['BOLD']

    table = [[
        f"{cpuTempColor}{str(cpu.temperature)}\N{DEGREE SIGN}C / {CtoF(cpu.temperature)}\N{DEGREE SIGN}F\n",
        f"{colors['OKGREEN']}{averageTemp} / {CtoF(averageTemp)}\N{DEGREE SIGN}F\n",
        f"{colors['OKGREEN']}{np.amax(temps)} / {CtoF(np.amax(temps))}\N{DEGREE SIGN}F\n",
        f"{colors['OKGREEN']}{np.amin(temps)} / {CtoF(np.amin(temps))}\N{DEGREE SIGN}F"
    ]]

    headers = [
        f"{colors['OKGREEN']}CPU TEMPERATURE",
        f"{colors['OKGREEN']}Average Temperature (last {minutes} minutes)",
        f"{colors['FAIL']}Peak Temperature (last {minutes} minutes)",
        f"{colors['OKCYAN']}Lowest Temperature (last {minutes} minutes){colors['OKGREEN']}", #OKGREEN at end is to make sure table lines are green, not cyan
    ]

    print('\n')
    fig = tpl.figure()
    plotConfig = {
        'width': width-2,
        'height': height-5,
        'label': 'CPU Temperature',
        'xlabel': 'Time (s)',
        'xlim': [times[0], times[-1:]],
        'ylim': [np.amin(temps)-2, np.amax(temps)+2],
        'title': f"CPU Temperature over last {minutes} minutes",
    }
    fig.plot(times, temps, **plotConfig)
    fig.show()
    # width=width-2, height=height-5, label='CPU Temperature', xlabel='Time (s)', , ylim=[np.amin(temps)-2, np.amax(temps)+2], title='CPU Temperature over last 5 minutes'
    print('\n')
    print(tabulate(table, headers=headers))

    time.sleep(tickRate)