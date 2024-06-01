import requests
import csv
import sys
import os
import json
from time_converter import date_weather_format, current_day_weather


def get_all_json_keys(keys_array, json):
    for key in json.keys():
        if not isinstance(json[key], str):
            _ = get_all_json_keys(keys_array, json[key][0])
        else:
            keys_array.append(key)
    return keys_array


def get_all_json_values(values_array, json):
    for key in json.keys():
        if not isinstance(json[key], str):
            _ = get_all_json_values(values_array, json[key][0])
        else:
            values_array.append(json[key])
    return values_array


# CONSTRUCTS THE API URL
def url_constructor(system):
    request_url = url
    request_url += "&q=" + system['lat_long']
    request_url += "&format=json"
    request_url += "&date=" + system['start_date']
    request_url += "&enddate=" + current_day_weather()
    request_url += "&includelocation=yes"
    request_url += "&tp=24"
    return request_url


if len(sys.argv) < 2:
    print("Please, inform the data path. EX: san_francisco")
    exit()

location_dir = sys.argv[1]


if not os.path.exists(location_dir+'/weathers'):
    os.mkdir(location_dir+'/weathers')

# RETRIEVES INFORMATION OF ALL FAVORITE SYSTEMS
system_list = []

with open(location_dir+"/favorite_systems.csv", "r") as systems:
    reader = csv.reader(systems, delimiter=",")
    next(reader)
    for line in reader:
        system_dict = {}
        system_dict['id'] = line[0]
        system_dict['start_date'] = date_weather_format(line[13])
        system_dict['end_date'] = "&enddate=" + current_day_weather()
        system_dict['lat_long'] = line[14] + "," + line[15]
        system_list.append(system_dict)

# ARRAY TO USE IN CASE PROGRAM FAILS IN THE MIDDLE BC OF LIMITED REQUESTS
ignore = ['52375', '29577', '55722', '70687', '8438', '41397', '13255',
          '54158', '72735', '65154', '176', '52412', '72288', '48885', '32239',
          '55434', '70830', '38742', '76398', '70775', '66542', '64779',
          '71919', '41921']

# BASE WEATHER API URL
url = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx?"
url += "key=912ce443276a4e86811154829220904"

for system in system_list:

    if system['id'] in ignore:
        continue
    print("Fetching weather data for system:", system['id'])
    weather_list = []
    while system['start_date'] != current_day_weather():
        request_url = url_constructor(system)
        response = requests.get(request_url)
        if response.status_code != 200:
            print(response.content)
            print(ignore)
            exit()

        content = response.content
        content = content.decode()
        content = json.loads(content)

        # REVERSES THE RESULT TO NEWST RECORDS ON TOP
        weather_list.insert(0, reversed(content['data']['weather']))

        system['start_date'] = content['data']['weather'][-1]['date']

    ignore.append(system['id'])

    # SAVES DATA INTO A FILE REFERING WITH THE SYSTEM ID
    file_name = location_dir + "/weathers/system_" + system['id'] + ".csv"
    with open(file_name, 'w') as test_daily:

        # FLAT THE JSON HEADERS
        header = ','.join(get_all_json_keys([], content['data']['weather'][0]))
        print(header, file=test_daily)

        # FLAT THE JSON VALUES
        for data_pack in weather_list:
            for daily_data in data_pack:
                row = ','.join(get_all_json_values([], daily_data))
                print(row, file=test_daily)
