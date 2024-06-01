# http://www.pythonchallenge.com/pc/def/peak.html

import requests, pickle

input_url = "http://www.pythonchallenge.com/pc/def/banner.p"
obj = requests.get(input_url)
text = obj.text
banner = pickle.loads(text)

final = []

for index, value in enumerate(banner):
    for j in value:
        final.append("".join(j[0] * j[1]))

    final.append('\n')

print "".join(final)
