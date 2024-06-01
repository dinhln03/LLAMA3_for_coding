#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from dve.io.table import TableDataBase
from jhunt.qt.widgets.mainwindow import MainWindow

import datetime

from PyQt5.QtWidgets import QApplication

APPLICATION_NAME = "JHunt"

def main():

    adverts_file_name = ".jhunt_adverts"

    adverts_data_schema = [
            {"header": "Date",         "default_value": datetime.datetime.now(), "dtype": datetime.datetime, "mapped": False},
            {"header": "Score",        "default_value": int(0),                  "dtype": int,               "mapped": False,  "min_value": 0,  "max_value": 5},
            {"header": "Application",  "default_value": False,                   "dtype": bool,              "mapped": False},
            {"header": "Category",     "default_value": "Entreprise",            "dtype": str,               "mapped": False,  "values": ("Entreprise", "IR/IE", "PostDoc")},
            {"header": "Organization", "default_value": "",                      "dtype": str,               "mapped": False},
            {"header": "Ref.",         "default_value": "",                      "dtype": str,               "mapped": False},
            {"header": "Title",        "default_value": "",                      "dtype": str,               "mapped": False},
            {"header": "URL",          "default_value": "",                      "dtype": str,               "mapped": True,  "widget": "QLineEdit"},
            {"header": "Pros",         "default_value": "",                      "dtype": str,               "mapped": True,  "widget": "QPlainTextEdit"},
            {"header": "Cons",         "default_value": "",                      "dtype": str,               "mapped": True,  "widget": "QPlainTextEdit"},
            {"header": "Description",  "default_value": "",                      "dtype": str,               "mapped": True,  "widget": "QPlainTextEdit"}
        ]
    
    adverts_database = TableDataBase(adverts_data_schema, adverts_file_name)

    websites_file_name = ".jhunt_websites"

    websites_data_schema = [
        {"header": "Date",         "default_value": datetime.datetime.now(), "dtype": datetime.datetime, "mapped": False,   "hidden": True},
        {"header": "Name",         "default_value": "",                      "dtype": str,               "mapped": False},
        {"header": "Score",        "default_value": int(0),                  "dtype": int,               "mapped": False,   "min_value": 0,   "max_value": 3},
        {"header": "Category",     "default_value": "Private Company",       "dtype": str,               "mapped": False,   "values": ("Private Company", "Public Research", "School", "Search Engine")},
        {"header": "Last visit",   "default_value": datetime.datetime.now(), "dtype": datetime.datetime, "mapped": False},
        {"header": "Today status", "default_value": "None",                  "dtype": str,               "mapped": False,   "values": ("None", "Partial", "Full")},
        {"header": "Description",  "default_value": "",                      "dtype": str,               "mapped": True,    "widget": "QPlainTextEdit"},
        {"header": "URL",          "default_value": "",                      "dtype": str,               "mapped": True,    "widget": "QLineEdit"}
    ]

    websites_database = TableDataBase(websites_data_schema, websites_file_name)

    adverts_data = adverts_database.load()              # TODO ?
    websites_data = websites_database.load()            # TODO ?

    app = QApplication(sys.argv)
    app.setApplicationName(APPLICATION_NAME)

    # Make widgets
    window = MainWindow(adverts_data, websites_data)

    # The mainloop of the application. The event handling starts from this point.
    # The exec_() method has an underscore. It is because the exec is a Python keyword. And thus, exec_() was used instead.
    exit_code = app.exec_()

    adverts_database.save(adverts_data)                # TODO ?
    websites_database.save(websites_data)              # TODO ?

    # The sys.exit() method ensures a clean exit.
    # The environment will be informed, how the application ended.
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
