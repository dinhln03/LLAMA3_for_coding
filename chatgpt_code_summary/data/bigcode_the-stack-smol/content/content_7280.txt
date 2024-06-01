from flask import Flask

app = Flask(__name__)
app.config.from_object('instapurge.settings')
app.secret_key = app.config['SECRET_KEY']

import instapurge.views
