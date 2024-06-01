from flask import Flask
from flask_restful import Api
from flask_cors import CORS
from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager

from {{cookiecutter.app_name}}.config import app_config
from {{cookiecutter.app_name}}.models import db, bcrypt
from {{cookiecutter.app_name}}.resources import Login, Register
from {{cookiecutter.app_name}}.schemas import ma


def create_app(env_name):
    """
    Create app
    """

    # app initiliazation
    app = Flask(__name__)
    CORS(app)

    app.config.from_object(app_config[env_name])

    # initializing bcrypt and db
    bcrypt.init_app(app)
    db.init_app(app)
    ma.init_app(app)
    migrate = Migrate(app, db)
    manager = Manager(app)
    manager.add_command('db', MigrateCommand)

    if __name__ == '__main__':
        manager.run()

    # Route
    api = Api(app)

    # user endpoint
    api.add_resource(Login, '/auth/login')
    api.add_resource(Register, '/auth/register')

    return app
