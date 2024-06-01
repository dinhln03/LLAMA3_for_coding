import os

from flask_script import Manager
from flask_migrate import MigrateCommand

from App import create_app

env = os.environ.get("flask_env", "develop")

app = create_app(env)


manager = Manager(app)
manager.add_command("db", MigrateCommand)

if __name__ == '__main__':
    manager.run()
