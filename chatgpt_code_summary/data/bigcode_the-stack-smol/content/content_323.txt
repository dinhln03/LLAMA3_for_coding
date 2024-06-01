"""
app.py - Flask-based server.

@author Thomas J. Daley, J.D.
@version: 0.0.1
Copyright (c) 2019 by Thomas J. Daley, J.D.
"""
import argparse
import random
from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
from wtforms import Form, StringField, TextAreaField, PasswordField, validators

from functools import wraps

from views.decorators import is_admin_user, is_logged_in, is_case_set

from webservice import WebService
from util.database import Database

from views.admin.admin_routes import admin_routes
from views.cases.case_routes import case_routes
from views.discovery.discovery_routes import discovery_routes
from views.drivers.driver_routes import driver_routes
from views.info.info_routes import info_routes
from views.login.login import login
from views.objections.objection_routes import objection_routes
from views.real_property.real_property_routes import rp_routes
from views.responses.response_routes import response_routes
from views.vehicles.vehicle_routes import vehicle_routes

from views.decorators import is_admin_user, is_case_set, is_logged_in

WEBSERVICE = None

DATABASE = Database()
DATABASE.connect()

app = Flask(__name__)

app.register_blueprint(admin_routes)
app.register_blueprint(case_routes)
app.register_blueprint(discovery_routes)
app.register_blueprint(driver_routes)
app.register_blueprint(info_routes)
app.register_blueprint(login)
app.register_blueprint(objection_routes)
app.register_blueprint(rp_routes)
app.register_blueprint(response_routes)
app.register_blueprint(vehicle_routes)


# Helper to create Public Data credentials from session variables
def pd_credentials(mysession) -> dict:
    return {
        "username": session["pd_username"],
        "password": session["pd_password"]
    }


@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')


@app.route('/attorney/find/<string:bar_number>', methods=['POST'])
@is_logged_in
def find_attorney(bar_number: str):
    attorney = DATABASE.attorney(bar_number)
    if attorney:
        attorney['success'] = True
        return jsonify(attorney)
    return jsonify(
        {
            'success': False,
            'message': "Unable to find attorney having Bar Number {}"
                       .format(bar_number)
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webservice for DiscoveryBot")
    parser.add_argument(
        "--debug",
        help="Run server in debug mode",
        action='store_true'
    )
    parser.add_argument(
        "--port",
        help="TCP port to listen on",
        type=int,
        default=5001
    )
    parser.add_argument(
        "--zillowid",
        "-z",
        help="Zillow API credential from https://www.zillow.com/howto/api/APIOverview.htm"  # NOQA
    )
    args = parser.parse_args()

    WEBSERVICE = WebService(args.zillowid)
    app.secret_key = "SDFIIUWER*HGjdf8*"
    app.run(debug=args.debug, port=args.port)
