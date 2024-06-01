"""
CSVLogger writes power values to a csv file.
"""
__author__ = 'Md Shifuddin Al Masud'
__email__ = 'shifuddin.masud@gmail.com'
__license__ = 'MIT License'

from pv_simulator.FileWriter import FileWriter
import csv
from datetime import datetime
import aiofiles
from aiocsv import AsyncWriter
import logging


class CSVFileWriter(FileWriter):
    __destination = ""

    def __init__(self, destination):
        """
        :param destination:
        """
        self.__destination = destination

    async def write(self, timestamp: datetime, meter_power_value: int, simulator_power_value: int,
                    combined_power_value: int) -> None:
        """
        Writes values into a csv file
        :param timestamp:
        :param meter_power_value:
        :param simulator_power_value:
        :param combined_power_value:
        :return:
        """
        async with aiofiles.open(self.__destination, mode='a') as csv_file:
            csv_file_writer = AsyncWriter(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            await csv_file_writer.writerow([datetime.now(), meter_power_value, simulator_power_value,
                                            combined_power_value])
        logging.debug("%s, %s, %s, %s are writen to %s", datetime.now(), meter_power_value, simulator_power_value,
                      combined_power_value, self.__destination)
