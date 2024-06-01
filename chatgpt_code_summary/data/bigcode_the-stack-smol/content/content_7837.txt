import json
import os
from time import sleep

import requests
import pyrominfo.pyrominfo.snes as snes
from shutil import copy

from pyrominfo.pyrominfo import nintendo64


def n64_info(filename):
    n64_parser = nintendo64.Nintendo64Parser()
    props = n64_parser.parse(filename)
    return props


def snes_info(filename):
    snes_parser = snes.SNESParser()
    props = snes_parser.parse(filename)
    return props


def get_console(argument):
    switcher = {
        'sfc': 'SNES',
        'smc': 'SNES',
        'md': '',
        'bin': '',
        'gb': 'GB',
        'gbc': 'GBC',
        'nes': 'NES',
        'z64': 'N64',
    }
    return switcher.get(argument)


def giant_bomb_request(title, api_key):
    headers = {'User-Agent': 'gripper'}
    params = {
        'resources': 'game',
        'query': title,
        'api_key': api_key,
        'format': 'json'
    }
    response = requests.get(url='http://www.giantbomb.com/api/search/', headers=headers, params=params)
    return json.loads(response.text)


def rip_game():
    while True:
        path = '/RETRODE'
        api_key = os.environ['api-key']

        files = os.listdir(path)
        files.remove('RETRODE.CFG')
        breakout = False
        console = get_console(files[0].split('.')[-1])
        filename = f'{path}/{files[0]}'
        if console == 'N64':
            rom_info = n64_info(filename)

        if console == 'SNES':
            rom_info = snes_info(filename)
            title = rom_info["title"]
            search_results = giant_bomb_request(title, api_key)
            for results in search_results['results']:
                if breakout is True:
                    break
                aliases = str(results.get('aliases')).lower().splitlines()
                if title.lower() in aliases or title.lower() == results['name']:
                    for platform in results['platforms']:
                        if platform['abbreviation'] == 'SNES':
                            if not os.path.exists(f'./{title}'):
                                os.mkdir(f'./{title} - {rom_info["region"]}')
                            for file in files:
                                destination_file = f'./{title} - {rom_info["region"]}/{title}.{file.split(".")[-1]}'
                                if not os.path.exists(destination_file):
                                    copy(filename, destination_file)
                            breakout = True
                            break
        sleep(5)

 #dont run code while testing container
if __name__ == '__main__':
    sleep(900)
    #rip_game()
