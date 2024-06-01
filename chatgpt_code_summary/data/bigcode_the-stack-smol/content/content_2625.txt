#!/usr/bin/python

import requests, time
from flask import Flask, Response, stream_with_context
app = Flask(__name__)

START = time.time()

def elapsed():
    running = time.time() - START
    minutes, seconds = divmod(running, 60)
    hours, minutes = divmod(minutes, 60)
    return "%d:%02d:%02d" % (hours, minutes, seconds)

@app.route('/<path:url>')
def root(url):
    req = requests.get("http://%s" % url, stream=True)
    return Response(stream_with_context(req.iter_content()), content_type = req.headers['content-type'])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
