#!/usr/bin/python3 -B
exec(open("../index.py").read())

from waitress import serve 
serve(application, host='0.0.0.0', port=8080, threads=1, channel_timeout=1) 
