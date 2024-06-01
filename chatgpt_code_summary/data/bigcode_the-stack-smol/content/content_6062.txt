#!venv/bin/python
""" This module imports Flask-Manager script, adds our create_db command
and run it. You can pass following arguments:
    * create_db => creates sqlite database and all the tables
    * shell => runs python shell inside application context
    * runserver => runs Flask development server
    * db => performs database migrations
        * db init => generate new migration
        * db migrate => generate automatic revision
        * db current => display current revision
        * db upgrade => upgrade to later version
        * db downgrade => revert to previous version
        * db history => list changes
        * db revision => create new revision file
        * db stamp => 'stamp' the revision table with giver revision

    optional arguments:
        -h, --help shows help message 
"""

from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.script import Manager, Command
from flask.ext.migrate import MigrateCommand
from app import app, db, migrate, models


class CreateDb(Command):
    """This class inherit from Flask-manager to add create_db command"""

    def run(self):
        """ Create database with all tables and print log to std.out"""
        print 'Creating the database.'
        db.create_all()

manager = Manager(app)
manager.add_command('db', MigrateCommand)
manager.add_command('create_db', CreateDb())

if __name__ == '__main__':
    manager.run()
