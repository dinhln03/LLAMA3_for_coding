# -*- coding: utf-8 -*-
from flask import Blueprint, jsonify

from flask_service.swagger import spec

__all__ = ['main_app']

main_app = Blueprint('main_app', __name__)


@main_app.route('/api')
def swagger():
    """
    Responds with the OpenAPI specification for this application.
    """
    return jsonify(spec.to_dict())


@main_app.route('/health')
def health():
    """
    Responds with the current's service health.

    Could be used by the liveness probe of a Kubernetes cluster for instance.
    """
    # put some logic here to decide if your app is doing well or not
    # by default, we'll always return everything is okay!
    return ""


@main_app.route('/status')
def status():
    """
    Responds with the current's service status.

    Could be used by the readiness probe of a Kubernetes cluster.
    """
    # put some logic here to decide if your app is doing well or not
    # by default, we'll always return everything is okay!
    return ""