import requests
import pprint

from config import API_KEY

base_url = f'https://api.telegram.org/bot{API_KEY}/'

api_response = requests.get(base_url + 'getUpdates').json()
for update in api_response['result']:

    message = update['message']
    chat_id = message['chat']['id']
    text    = message['text']
    reply_message = {
        'chat_id': chat_id,
        'text': text
    }
    requests.post(base_url + 'sendMessage', json=reply_message)

# pprint.pprint(api_response['result'][0])