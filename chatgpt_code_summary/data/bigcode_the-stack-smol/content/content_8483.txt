import requests

def hello():
	response = requests.get('http://weather.livedoor.com/forecast/webservice/json/v1?city=130010')
	weather = response.json()["forecasts"][0]["telop"]
	return 'Hello, the weather in tokyo today is ' + weather

