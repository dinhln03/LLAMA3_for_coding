import inspect
import os
import pyperclip
import requests
import time
from urllib.parse import quote

# a list of the request error classes
request_errors = [obj for name, obj in inspect.getmembers(requests.exceptions)
                  if inspect.isclass(obj) and issubclass(obj, Exception)]
# main daemon loop
while True:
    # get clipboard value
    clipboard = pyperclip.paste()
    try:
        # percent encode the clipboard value
        safe_cb = quote(clipboard,safe='')
        # bitly API access token
        token = os.environ.get('BITLY_TOKEN')
        # URL that will make the API call
        bitly_url = 'https://api-ssl.bitly.com/v3/shorten?' + \
                    'access_token=' + token + '&longUrl=' + safe_cb
        # get the json return from the API call
        short_url = requests.get(bitly_url).json()
        # if everything went as planned
        if(short_url['status_txt'] == 'OK'):
            pyperclip.copy(short_url['data']['url'])
    except Exception as e:
        # if something went wrong with the request, i.e. not a link
        if(any(issubclass(e.__class__, lv) for lv in request_errors)):
            pass
        else:
            raise(e)
    # wait until the clipboard changes
    while(pyperclip.paste() == clipboard):
        time.sleep(.1)
