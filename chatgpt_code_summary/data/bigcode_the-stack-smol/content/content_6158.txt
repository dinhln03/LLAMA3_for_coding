# Proton JS - Proton.py
# by Acropolis Point

# module imports
import os
import json
import time
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

@app.route('/new', methods=['POST'])

# new() function definition
def new():
    os.system("python3 window.py " + request.get_data(as_text = True))
    return 'OK'

@app.route('/shell', methods=['POST'])

# shell() function definition
def shell():
    os.system(request.get_data(as_text = True))
    return 'OK'

@app.route('/filesave', methods=['POST'])
def filesave():
    theFile = open(request.get_data(as_text = True).split(", ")[1], "w+")
    theFile.write(request.get_data(as_text = True).split(", ")[0])
    return 'OK'

@app.route('/close', methods=['POST'])
def close():
    theFile = open("output.json", "r+")
    theFileParsed = json.load(theFile)
    theFileParsed['close'] = request.get_data(as_text = True)
    theFile.seek(0)
    theFile.write(json.dumps(theFileParsed) + "      ")
    time.sleep(200)
    theFile.write("{ \"close\": \"\" }")
    return 'OK'
