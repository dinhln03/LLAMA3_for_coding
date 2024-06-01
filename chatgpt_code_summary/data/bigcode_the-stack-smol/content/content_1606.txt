# The Core of Toby
from flask import Flask, request, jsonify, g
import os
import logging
from ax.log import trace_error
from ax.connection import DatabaseConnection
from ax.datetime import now
from ax.tools import load_function, get_uuid, decrypt
from ax.exception import InvalidToken


logger = logging.getLogger('werkzeug')
debug_flg = True if os.getenv('TOBY_DEBUG', 'True') == 'True' else False
token = os.environ['TOBY_TOKEN']
app = Flask('Toby')
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.logger.setLevel(logging.DEBUG if debug_flg else logging.INFO)


def get_db():
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(g, 'db'):
        g.db = DatabaseConnection(os.getenv('TOBY_DB_USER', 'toby'), os.environ['TOBY_DB_PASSWORD'])
    return g.db


@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if hasattr(g, 'db'):
        g.db.disconnect()
        if error:
            logger.error('Database connection closed because of :' + str(error))


@app.route("/")
def ping():
    return "<h1 style='color:blue'>Hello There! This is Toby</h1>"


@app.route("/process")
def process():
    request_id = None
    try:
        in_param = request.get_json(force=True, silent=False, cache=False)
        if decrypt(in_param['request_token']) != token:
            # verify token
            raise InvalidToken(in_param)
        if 'request_id' not in in_param:
            request_id = get_uuid()
            in_param['request_id'] = request_id
        else:
            request_id = in_param['request_id']
        if 'request_timestamp' not in in_param:
            in_param['request_timestamp'] = now()
        in_param['logger'] = logger
        in_param['get_db_connection'] = get_db
        func = load_function(in_param)
        resp = func()
    except:
        e = trace_error(logger)
        resp = {'request_id': request_id, 'request_status': 'error', 'request_error': str(e[-1])}
    return jsonify(resp)


if __name__ == "__main__":
    app.run()
