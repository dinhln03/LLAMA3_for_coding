import json

with open('history.json') as jsondata:
    x = json.load(jsondata)

if x:
    x = "".join(x)
    with open('history.html', 'r') as html:
        html_data = html.read()
    html_data = html_data.replace("%s", x)
    with open('temp.html', 'w') as html:
        html.write(html_data)
else:
    with open('history.html', 'r') as html:
        html_data = html.read()
    html_data = html_data.replace("%s", "")
    with open('temp.html', 'w') as html:
        html.write(html_data)
