# coding=utf-8
import requests
import json

def robot(content,userid):
    api = r'http://openapi.tuling123.com/openapi/api/v2'
    data = {
        "perception": {
            "inputText": {
                "text": content
                         }
                      },
        "userInfo": {
                    "apiKey": "fece0dcdbe4845559492c26d5de40119",
                    "userId": userid
                    }
    }
    response = requests.post(api, data=json.dumps(data))
    robot_res = json.loads(response.content)
    return robot_res["results"][0]['values']['text']

