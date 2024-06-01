import json
import time
import datetime
import requests
import json

playbackStampRecords = [{"b500c6b0-633b-11ec-85c5-cba80427674d2021-10-10"}]

def getConfigurations():
    url = "https://data.mongodb-api.com/app/data-mtybs/endpoint/data/beta/action/find"
    payload = json.dumps({
        "collection": "configurations",
        "database": "themorningprojectdb",
        "dataSource": "Cluster0"
    })
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Request-Headers': '*',
        'api-key': 'kvErM5pzFQaISsF733UpenYeDTT7bWrJ85mAxhz956wb91U5igFxsJoDEDpyW6NJ'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    jsonResponse = response.json()

    return jsonResponse['documents']


def isNotificationScheduled():
    configList = getConfigurations()
    currentDate = datetime.datetime.now().date()
    currentTime = datetime.datetime.now().strftime("%H:%M")
    for config in configList:
        if config["id"]+str(currentDate) not in playbackStampRecords and currentTime >= config['settings'][0]['starttime'] and currentTime <= config['settings'][0]['endtime']:
            return True

def getUserConfiguration():
    configList = getConfigurations()
    currentDate = datetime.datetime.now().date()
    currentTime = datetime.datetime.now().strftime("%H:%M")

    for config in configList:
        if config["id"]+str(currentDate) not in playbackStampRecords and currentTime >= config['settings'][0]['starttime'] and currentTime <= config['settings'][0]['endtime']:
            email = config['settings'][0]['amemail']
            password = config['settings'][0]['ampassword'] 
            name = config['settings'][0]['name']
            playbackInformation = config['settings'][0]['info'] 
            if currentTime >= '00:00'and currentTime < '12:00':
                dayTimeDescriptor = 'morning'
            elif currentTime >= '12:00' and currentTime <= '16:00':
                dayTimeDescriptor = 'afternoon'
            else: 
                dayTimeDescriptor = 'night'   
            playbackStampRecords.append(config['id']+str(currentDate))
            return [email, password, dayTimeDescriptor, name, playbackInformation]


