import requests
import json
from pprint import pprint as print

def getCode(res:str) :
    return str(res).split("[")[1].split("]")[0]

url = 'http://localhost:4042'
guid = '2012491924' # get guid from connexion.json()
guid2 = '0'
gurl = f"{url}/{guid}"

home = requests.post(url)
print (getCode(home))
print (home.json())
print ("\n\n##################\n\n")

connexion = requests.post('http://localhost:4042/connect')
print (getCode(connexion))
print (connexion.json())
print ("\n\n##################\n\n")

# regarder =  requests.get(f"{gurl}/regarder")
# print (getCode(regarder))
# print (regarder.json())
# print ("\n\n##################\n\n")

# myobj = {"direction": "N"}
# deplacement =  requests.post(f"{gurl}/deplacement", json=myobj)
# print (getCode(deplacement))
# print (deplacement.json())
# print ("\n\n##################\n\n")

# examiner =  requests.get(f"{gurl}/examiner/{guid2}")
# print (getCode(examiner))
# print (examiner.json())
# print ("\n\n##################\n\n")

# taper =  requests.get(f"{gurl}/taper/{guid2}")
# print (getCode(taper))
# print (taper.json())

