import requests
#Using an API to search for COVID Movies
url = 'http://www.omdbapi.com/?t=covid&y=2021&apikey=d0c69d2c'
r = requests.get(url)
json_data = r.json()
for key, value in json_data.items():
print(key + ':', value)