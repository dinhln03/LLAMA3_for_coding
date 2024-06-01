def path_hack():
    import os, sys, inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir) 
    # print('path added:', sys.path[0])

path_hack()

import traceback
import sys
import urllib.request
from urllib.request import urlopen
import json
from apis import utilities

try:
    from apis import my_token
    API_TUTOR_TOKEN = my_token.API_TUTOR_TOKEN
except:
    title = 'IMPORTANT: You Need an Access Token!'
    error_message = '\n\n\n' + '*' * len(title) + '\n' + \
        title + '\n' + '*' * len(title) + \
        '\nPlease download the the my_token.py file and save it in your apis directory.\n\n'
    raise Exception(error_message)



def get_token(url):
    try:
        response = urlopen(url + '?auth_manager_token=' + API_TUTOR_TOKEN)
        data = response.read()
        results = data.decode('utf-8', 'ignore')
        return json.loads(results)['token']
    except urllib.error.HTTPError as e:
        # give a good error message:
        error = utilities.get_error_message(e, url)
    raise Exception(error)