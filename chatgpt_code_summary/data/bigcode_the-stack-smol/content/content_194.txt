from fake_useragent import UserAgent
import requests
from jsonpath import jsonpath

url = "https://www.lagou.com/lbs/getAllCitySearchLabels.json"

headers = {"User-Agent": UserAgent().chrome}
resp = requests.get(url, headers=headers)


ids = jsonpath(resp.json(), "$..id")
names = jsonpath(resp.json(), "$..name")

for id, name in zip(ids, names):
    print(id, ":", name)
