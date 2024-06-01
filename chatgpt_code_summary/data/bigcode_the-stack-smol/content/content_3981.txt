"""App initialization file.  Instantiates app, database, login_manager.  Registers view blueprints.  Defines user_loader callback for LoginManager."""

from flask import Flask
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

from database import init_db, session
from models import Base, Category, Item, User
from views.auth import authModule
from views.categories import categoryModule
from views.items import itemModule
from views.site import siteModule

login_manager = LoginManager()

app = Flask(__name__)

login_manager.init_app(app)

csrf = CSRFProtect(app)

init_db()


@login_manager.user_loader
def load_user(userid):
    user = session.query(User).filter_by(id=userid).first()
    print "Trying to load %s" % user
    if user:
        return user
    else:
        return None


@app.teardown_appcontext
def shutdown_session(exception=None):
    session.remove()


app.register_blueprint(categoryModule)
app.register_blueprint(itemModule)
app.register_blueprint(authModule)
app.register_blueprint(siteModule)
