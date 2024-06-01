import os
from connexion import App
from flask_marshmallow import Marshmallow
from flask_sqlalchemy import SQLAlchemy

basedir = os.path.abspath(os.path.dirname(__file__))

conn = App(__name__, specification_dir='./')

app = conn.app

postgres_url = 'postgres://postgres:docker@10.5.95.65:54320/web_service_db'

app.config["SQLALCHEMY_ECHO"] = True
app.config["SQLALCHEMY_DATABASE_URI"] = postgres_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = basedir + os.sep + "web_service_files"
app.config["DATABASE"] = "web_service_db"
app.config["PORT"] = 5433
app.config["USERNAME"] = "postgres"
app.config["HOSTNAME"] = "10.5.95.65"

db = SQLAlchemy(app)

ma = Marshmallow(app)


