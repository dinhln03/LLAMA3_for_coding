#!/usr/bin/env python

import csv
import os
import argparse
import dateutil.parser
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, required=True,
                        help="name of the data directory")
    args = parser.parse_args()
    return args.dir

def convert_ts(ts_str):
    return dateutil.parser.parse(ts_str).timestamp()

def get_data(data, fname):
    with open(fname, newline='') as csvfile:
        content = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in content:
            if len(line) < 5:
                continue
            try:
                ts = convert_ts(line[2])

                adm = [line[1]]
                if line[0] != '':
                    adm.append(line[0])
                
                data.append(
                    {
                        'date': ts,
                        'adm': adm,
                        'infected': int(line[3]),
                        'deaths': int(line[4]),
                        'recovered': int(line[5]),
                        'sex': 'NaN', # Not sure why this is needed????
                        # 'source': 'JHU',
                        'source': ObjectId("5e75f8d7745bde4a48972b42")
                    })
            except ValueError as ve:
                # If there is a problem e.g. converting the ts
                # just go on.
                pass

def convert2json(dir_name):
    data = []

    for fname in os.listdir(dir_name):
        get_data(data, os.path.join(dir_name, fname))

    return data

def main():
    dir_name = parse_args()
    data = convert2json(dir_name)
    print(json.dumps(data))

if __name__ == '__main__':
    main()
