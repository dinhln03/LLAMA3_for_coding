import random
from string import Template

import tinydb
from flask import Flask, redirect, url_for, abort

app = Flask(__name__)
db = tinydb.TinyDB('./db.json')

HTML = """<!DOCTYPE html>
<html>
  <head>
    <title>Dulux Swatch</title>
    <link rel="StyleSheet" href="/static/main.css" type="text/css">
  </head>
  <body style="background-color: #${rgb}">
    <div>
        ${name}
        <br/>
        (id: <a href="https://www.dulux.co.uk/en/products/colour-tester#?selectedColor=${colorId}">${colorId}</a>)
    </div>
  </body>
</html>
"""

@app.route('/id/<colour_id>')
def get_colour_by_id(colour_id):
    obj = db.get(tinydb.Query().colorId == colour_id)
    if not obj:
        abort(404)
    return Template(HTML).substitute(obj)


@app.route('/name/<colur_name>')
def get_colour_by_name(colur_name):
    obj = db.get(tinydb.Query().uriFriendlyName == colur_name)
    if not obj:
        abort(404)
    return Template(HTML).substitute(obj)


@app.route('/')
@app.route('/random')
def get_random_colour():
    return redirect(url_for('get_colour_by_id', colour_id=random.choice(db.all())['colorId']))
