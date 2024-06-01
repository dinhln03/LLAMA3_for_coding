import glob
import pandas as pd
import os
import datetime


class DataMerge:
    def __init__(self, directory):
        self.directory = directory
        self.__data = self.get_data_from(self.directory)

    def date_to_int(self, dates):
        """
        calculates number of days between 01/01/0001 and each date in dates
        date has format '%m/%d/%Y'

        :param dates: Pandas Series
        :return: list
        """
        ret = []
        for date in dates:
            date0 = datetime.datetime(year=1, month=1, day=1)
            datex = datetime.datetime.strptime(date, '%m/%d/%Y')
            ret.append((datex - date0).days)
        return ret

    def get_data_from(self, dir):
        files = glob.glob(f'{dir}/*')
        if files == []:
            raise f'directory {dir} does not contain any .csv file'
        data = None
        for file in files:
            if file == f'{dir}/merged_data.csv':
                continue
            if data is None:
                data = pd.read_csv(file)
                continue
            temp_data = pd.read_csv(file)
            temp_data = temp_data.dropna(axis=1)
            data = data.append(temp_data)
        data.drop_duplicates()
        data = data.sort_values('Date', ascending=False, key=self.date_to_int)
        data = data[: 408]
        data.to_csv(f"{dir}/merged_data.csv", index=False)
        return data

    def get_data(self):
        return self.__data