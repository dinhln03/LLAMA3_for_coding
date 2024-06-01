"""
module init
"""
from flask import Flask
<<<<<<< HEAD
from config import config_options
from flask_sqlalchemy import SQLAlchemy
import os
=======
from config import DevelopmentConfig
from .views import orders_blue_print
>>>>>>> ba86ec7ade79a936b81e04ee8b80a97cf8f97770


def create_app(DevelopmentConfig):
    """
    Function create_app:
    creates app and gives it the import name
    holds the configuration being used.
    registers the orders blueprint
    :return: app:
    """
    app = Flask(__name__)
    app.config.from_object(DevelopmentConfig)
    app.register_blueprint(orders_blue_print)

<<<<<<< HEAD
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # set the configurations
    app.config.from_object(os.environ['APP_SETTINGS'])
    db=SQLAlchemy(app)

    # initialiaze the database
    db.init_app(app)


    with app.app_context():
        from .import routes
        db.create_all
    # register your blueprints here
    from app.main import main
    from app.auth import auth
    

    app.register_blueprint(main)
    app.register_blueprint(auth)


    @app.route('/')
    def  hello():
        return "Hello World!"


    return app
=======
    return app 
>>>>>>> ba86ec7ade79a936b81e04ee8b80a97cf8f97770
