import time
import os

from flask import Flask, jsonify, make_response
from flask.ext.sqlalchemy import SQLAlchemy
from redis import Redis
from rq import Queue
from fetch import fetch_user_photos


app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
db = SQLAlchemy(app)
request_queue = Queue(connection=Redis())

from models import Profile

@app.route("/")
def index():
    return jsonify({
        "msg": "Welcome to PyPhotoAnalytics",
        "routes": ["/api", "/api/users", "/api/users/<username>"]
    })

@app.route("/api")
def api():
    return jsonify({"msg": "Welcome to PyPhotoAnalytics API"})

@app.route("/api/users/")
def get_users():
    return jsonify({"msg": "specify username /api/users/<username>"})

@app.route("/api/users/<username>")
def get_user_media(username):
    job = request_queue.enqueue(fetch_user_photos, username)
    time.sleep(7)

    result = job.result
    if result is None:
        return jsonify({"msg": "Still processing :("})
    elif result.status_code == 200:
        data = result.json()
        return jsonify(**data)
    else:
        return jsonify({"msg": "Oh gawd no"})

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": "Not Found"}), 404)

if __name__ == "__main__":
    app.run()
