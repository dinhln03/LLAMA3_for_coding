import urllib.request as ul
import json
import pandas as pd


def get_chart(ticker, period1, period2):
    url = f"http://localhost:9000/chart/{ticker}?period1={period1}&period2={period2}"

    request = ul.Request(url)
    response = ul.urlopen(request)

    rescode = response.getcode()
    if rescode != 200:
        return None

    responsedata = response.read()
    my_json = responsedata.decode('utf8').replace("'", '"')
    data = json.loads(my_json)

    return data["data"]["history"]


info = get_chart("aaa", 20211015, 20211104)
df = pd.json_normalize(info)
df.to_csv("aaa_chart.csv")

print(df)
