#!/usr/bin/python3

import importlib
import os
import getpass
import pip

from crontab import CronTab

if int(pip.__version__.split('.')[0])>9:
    from pip._internal import main as pipmain
else:
    from pip import main as pipmain


def check_modules():

    packages = {"docx" : "python-docx",
                "googleapiclient" : "google-api-python-client",
                "google_auth_oauthlib" : "google_auth_oauthlib",
                "crontab" : "python-crontab"
            }

    for ky, vl in packages.items():
        spam_spec = importlib.util.find_spec(ky)

        if spam_spec is None:
            pipmain(['install', vl])


class useCronTab(object):

    def __init__(self):

        usr_name = getpass.getuser()
        self.my_cron = CronTab(user=usr_name)

        abs_path = os.path.dirname(os.path.realpath(__file__))
        self.commnd = abs_path + "src/agenda "


    def set_job(self, date, featur):
        job = self.my_cron.new(command=self.commnd+featur)

        job.setall(date)


    def __del__(self):
        self.my_cron.write()


def import_package():
    import py_compile

    py_modules = ["cal_setup.py",
                  "classEvent.py",
                  "datesGenerator.py",
                  "intrctCalendar.py",
                  "intrctDrive.py",
                  "readDocxFile.py",
                  "getNotification.py"
                 ]

    for pckg in py_modules:
        py_compile.compile(pckg)


if __name__ == '__main__':

    abs_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(abs_path)

    print("Installing missing modules")
    check_modules()
    print("Compiling .py files")
    import_package()

    features = {"today" : "-d",
                "tomorrow" : "-t",
                "week" : "-w",
                "month" : "-m"
               }

    automatize = input("Do you want to automitize executation? [y]/[n]: ")

    cron = useCronTab()

    while automatize == 'y':
        feature = input("Dates to update automaticaly? [today]/[tomorrow]/[week]/[month]: ")

        print("\n Select the day: \n"
              + "-----------------------\n"
              + "dow: day of week (0-6 Sun-Sat)\n"
              + "mon: Month (1-12 Jan-Dec)\n"
              + "dom: day of month (1-31)\n"
              + "hh: hour (00-23)\n"
              + "mm: minute (00-59)\n"
              + "Use * for any\n")

        date = input("Select date [mm hh dom mon dow]: ")

        cron.set_job(date, features[feature])

        automatize = input("Any more automatization? [y]/[n]:")

    del cron
