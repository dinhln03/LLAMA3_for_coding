#!/usr/bin/python
import os
import json

def get_db_config():
    # read config file and return data
    data = {}
    with open('config.json', 'r') as infile:
        data = json.loads(infile.read())
    return data
