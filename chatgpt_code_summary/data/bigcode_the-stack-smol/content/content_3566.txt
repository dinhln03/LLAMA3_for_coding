from Utilities.CSV import csv_data_line
from Utilities import date_formating
import logging
from datetime import date
import time
import datetime

from shared_types import DateDict

logging.basicConfig(filename='../../CrawlerLogs' + 'Crawlerlog-' +
                    date.today().strftime("%b-%Y") + '.log',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')


def process_file(filename: str) -> DateDict:
    """
    Method that take path to crawled file and outputs date dictionary:
    Date dictionary is a dictionary where keys are dates in format YYYY-mm-dd-hh (2018-04-08-15)
    and value is dictionary where keys are devices (specified in configuration file)
    and value is CSVDataLine.csv_data_line with device,date and occurrence

    Args:
    filename: name of processed file

    Returns:
    None if not implemented
    date_dict when implemented
    """
    date_dict = {}

    with open(filename, "r") as file:

        YEAR_START = 1
        YEAR_END = 11
        for line in file:

            array = line.split(";")

            #pick later time
            time_ = max(
                array[2][1:-1],
                array[3][1:-1],
                key=lambda x: time.mktime(
                    datetime.datetime.strptime(x, "%H:%M").timetuple()))

            date = date_formating.date_time_formatter(
                array[14][YEAR_START:YEAR_END] + " " + time_)

            name = array[10][1:-1]
            if name == "":
                continue

            if date not in date_dict:
                date_dict[date] = {}

            if name in date_dict[date]:
                date_dict[date][name].occurrence = int(array[12])
            else:
                date_dict[date][name] = csv_data_line.CSVDataLine(
                    name, date, int(array[12]))

    return date_dict
