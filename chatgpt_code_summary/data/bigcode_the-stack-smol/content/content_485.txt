""" Third party api wrappers"""
import os
import json
import nexmo
import africastalking

username = os.getenv('africastalking_username')
api_key = os.getenv('africastalking_api_key')
africastalking.initialize(username, api_key)
sms = africastalking.SMS


class ProvidersWrapper:
    """ Class with all the thirdy party helper functions"""

    def send_message(number, message):
        client = nexmo.Client(key=os.getenv('nexmokey'), secret=os.getenv('nexmosecret'))
        response = client.send_message({
            'from': 'Nexmo',
            'to': number,
            'text': message,
            })
        if response["messages"][0]["status"] != "0":
            response = sms.send(message, ['+' + number])
        return response

