import logging
import pandas as pd
from flask import Flask, request
from gevent.pywsgi import WSGIServer
from time import sleep

from func import rms, meas_to_influx, rms_to_influx, config

logger = logging.getLogger(config['log_name'])
logger.setLevel(logging.INFO)
h_stream = logging.StreamHandler()
h_stream.setLevel(logging.INFO)
logger.addHandler(h_stream)

app = Flask(__name__)


@app.post('/save')
def save():
    headers = request.headers
    if 'X-API-KEY' not in headers or headers['X-API-KEY'] != config['api_key']:
        sleep(5)
        return '', 401

    data = request.json
    dt = pd.Timestamp(data['dt'])
    s_data, power = rms(data['payload'], data['ticks'], dt)

    if power < 0:
        logger.error(data)
        return '', 204

    if power < 100:
        return str(power)

    # print(s_data)
    # print(power)
    rms_to_influx(power, dt)
    meas_to_influx(s_data)

    return str(power)


if __name__ == '__main__':
    # app.run(host=config['url'], port=config['port'])
    WSGIServer((config['url'], config['port']), app).serve_forever()
