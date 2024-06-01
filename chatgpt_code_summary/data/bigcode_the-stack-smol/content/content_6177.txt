# -----------------------------------------------------
# File: temperature.py
# Author: Tanner L
# Date: 09/20/19
# Desc: Temperature sensor communication
# Inputs:
# Outputs: temperature
# -----------------------------------------------------
import threading as th
import logging
import time
import interface
import adafruit_dht  # import library for temperature sensor
import board


# -----------------------------------------------------
# Function: class - temperatureThread
# Author: Tanner L
# Date: 10/10/19
# Desc: Adjusts gps values based on settings for display
# Inputs:
# Outputs:
# -----------------------------------------------------
class TemperatureThread(th.Thread):
    def __init__(self):
        th.Thread.__init__(self)

        logging.info('--------------------TEMPERATURE START----------------------------')
        self.go = True

    # -----------------------------------------------------
    # Function: run
    # Author: Tanner L
    # Date: 10/10/19
    # Desc: Loop for temperatureThread, gets temperature from sensor
    # Inputs:
    # Outputs:
    # -----------------------------------------------------
    def run(self):

        sensor = adafruit_dht.DHT11(board.D16)  # setup dht11 to be read

        while self.go:

            try:
                interface.temperature_queue = sensor.temperature  # read in temperature

            except:
                logging.error('Temperature Sensor Error')
                print('Temp Read Error')

            if interface.temperature_queue <= 0:
                time.sleep(1)  # send new temperature every 1 seconds
            else:
                time.sleep(10)

    # -----------------------------------------------------
    # Function: stop_thread
    # Author: Tanner L
    # Date: 10/10/19
    # Desc: Stops thread for shutdown
    # -----------------------------------------------------
    def stop_thread(self):  # used to kill thread
        self.go = False
