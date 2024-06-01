#!/usr/bin/env python3

self_description = """
gridradar2influx is a tiny daemon written to fetch data from the gridradar.net-API and
writes it to an InfluxDB instance.
"""

# import standard modules
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import configparser
import logging
import os
import signal
import time
from datetime import datetime

# import 3rd party modules
import requests
import influxdb
#import functions from files
from app_functions import *
from basic_functions import *
from influx import *

__version__ = "0.0.1"
__version_date__ = "2022-02-05"
__description__ = "gridradar2influx"
__license__ = "MIT"

# default vars
running = True
default_config = os.path.join(os.path.dirname(__file__), 'config.ini')
default_log_level = logging.INFO








def main():
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    # parse command line arguments
    args = parse_args()
    # set logging
    log_level = logging.DEBUG if args.verbose is True else default_log_level
    if args.daemon:
        # omit time stamp if run in daemon mode
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s: %(message)s')
    # read config from ini file
    config = read_config(args.config_file)
    # set up influxdb handler
    influxdb_client = None
    try:
        influxdb_client = influxdb.InfluxDBClient(
            config.get('influxdb', 'host'),
            config.getint('influxdb', 'port', fallback=8086),
            config.get('influxdb', 'username'),
            config.get('influxdb', 'password'),
            config.get('influxdb', 'database'),
            config.getboolean('influxdb', 'ssl', fallback=False),
            config.getboolean('influxdb', 'verify_ssl', fallback=False)
        )
        measurement_name=config.get('influxdb', 'measurement_name')
        location=config.get('influxdb', 'location')
        # test more config options and see if they are present
        #_ = config.get('influxdb', 'measurement_name')
    except configparser.Error as e:
        logging.error("Config Error: %s", str(e))
        exit(1)
    except ValueError as e:
        logging.error("Config Error: %s", str(e))
        exit(1)
    # check influx db status
    check_db_status(influxdb_client, config.get('influxdb', 'database'))

    # create authenticated gridradar-api client handler
    api_response = None
    result_dict={}
    request_interval = 60
    try:
        request_interval = config.getint('gridradar', 'interval', fallback=60)
        url=config.get('gridradar', 'url')
        token=config.get('gridradar', 'token')
        api_response=getdatafromapi(url,token,{}) # blank request to check, if authentification works

 
    except configparser.Error as e:
        logging.error("Config Error: %s", str(e))
        exit(1)
    except BaseException as e:
        logging.error("Failed to connect to gridradar-API  '%s'" % str(e))
        exit(1)

    # test connection
    try:
        api_response
    except requests.exceptions.RequestException as e:
        if "401" in str(e):
            logging.error("Failed to connect to gridradar-API '%s' using credentials. Check token!" %
                          config.get('gridradar', 'token'))
        if "404" in str(e):
            logging.error("Failed to connect to gridradar-API '%s' using credentials. Check url!" %
                          config.get('gridradar', 'url'))
        else:
            logging.error(str(e))
        exit(1)

    logging.info("Successfully connected to gridradar-API")
    # read services from config file
    ###services_to_query = get_services(config, "service")
    logging.info("Starting main loop - wait until first API-Request '%s' seconds",request_interval)

    while running:
        logging.debug("Starting gridradar-API requests")
        time.sleep(request_interval) # wait, otherwise Exception 429, 'Limitation: maximum number of requests per second exceeded']
        
        request=str2dict(config.get('gridradar', 'request_freq'))
        duration=grapi2influx(request,influxdb_client,config)
        
        # just sleep for interval seconds - last run duration
        for _ in range(0, int(((request_interval * 1000) - duration) / 100)):
            if running is False:
                break
            time.sleep(0.0965)
        
        request=str2dict(config.get('gridradar', 'request_net_time'))
        duration=grapi2influx(request,influxdb_client,config)
        # just sleep for interval seconds - last run duration
        for _ in range(0, int(((request_interval * 1000) - duration) / 100)):
            if running is False:
                break
            time.sleep(0.0965)
            
            
if __name__ == "__main__":
    main()
