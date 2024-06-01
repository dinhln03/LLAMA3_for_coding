import os
import json


__author__ = 'Manfred Minimair <manfred@minimair.org>'


class JSONStorage:
    """
    File storage for a dictionary.
    """
    file = ''  # file name of storage file
    data = None  # data dict
    indent = '  '  # indent prefix for pretty printing json files

    def __init__(self, path, name):
        """
        Initizlize.
        :param path: path to the storage file;
        empty means the current direcory.
        :param name: file name, json file; may include a path.
        """
        if path:
            os.makedirs(path, exist_ok=True)
        self.file = os.path.normpath(os.path.join(path, name))
        try:
            with open(self.file) as data_file:
                self.data = json.load(data_file)
        except FileNotFoundError:
            self.data = dict()
            self.dump()

    def dump(self):
        """
        Dump data into storage file.
        """
        with open(self.file, 'w') as out_file:
            json.dump(self.data, out_file, indent=self.indent)

    def get(self, item):
        """
        Get stored item.
        :param item: name, string, of item to get.
        :return: stored item; raises a KeyError if item does not exist.
        """
        return self.data[item]

    def set(self, item, value):
        """
        Set item's value; causes the data to be dumped into the storage file.
        :param item: name, string of item to set.
        :param value: value to set.
        """
        self.data[item] = value
        self.dump()

    def __getattr__(self, item):
        """
        Get stored item with .-notation if not defined as a class member.
        :param item: name, string of item compatible
        with Python class member name.
        :return value of item.
        """
        if item in self.data:
            return self.data[item]
        else:
            raise AttributeError
