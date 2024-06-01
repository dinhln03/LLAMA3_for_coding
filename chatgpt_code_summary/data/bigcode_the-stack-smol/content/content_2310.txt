import json
import os
import pickle
import requests
import shutil
import tempfile
import uuid

from flask import Blueprint, current_app, jsonify, request, send_file


name = 'HTTP'
prefix = 'http'
storage_enabled = True
global storage_path

plugin = Blueprint(name, __name__)


def register(app, plugin_storage_path=None):
    app.register_blueprint(plugin, url_prefix=f'/{prefix}')
    app.logger.info(f'{name} plugin registered.')
    global storage_path
    storage_path = plugin_storage_path


persistence = {
    "configuration": {},
    "execution": {},
}

result_zip_file_name = 'results.zip'


@plugin.route('/')
def index():
    return f'This is the Radon CTT Agent HTTP Plugin.', 200


@plugin.route('/configuration/', methods=['POST'])
def configuration_create():
    config_instance = {}
    configuration_uuid = str(uuid.uuid4())
    config_instance['uuid'] = configuration_uuid

    params = {
        'use_https': {
            'required': True,
            'default': False,
        },
        'method': {
            'required': True,
            'default': 'GET',
        },
        'hostname': {
            'required': True,
            'default': None,
        },
        'port': {
            'required': True,
            'default': 80,
        },
        'path': {
            'required': True,
            'default': "/",
        },
        'test_body': {
            'required': False,
            'default': None,
        },
        'test_header': {
            'required': False,
            'default': None,
        },
    }

    for param in params:
        is_required = params[param]['required']
        default_value = params[param]['default']

        if param in request.form:
            value = request.form.get(param, type=str)
            current_app.logger.info(f'\'{param}\' set to: \'{value}\'.')
            config_instance[param] = value
        else:
            if is_required and default_value is not None:
                value = default_value
                current_app.logger.info(f'\'{param}\' set to default value: \'{value}\'.')
                config_instance[param] = value

        if is_required and param not in config_instance:
            current_app.logger.error(f"Required parameter {param} not provided.")
            return f'Required parameter {param} not provided.', 400

    persistence['configuration'][configuration_uuid] = config_instance
    current_app.logger.info(f"Config: {config_instance}")
    return jsonify(config_instance), 201


@plugin.route('/execution/', methods=['POST'])
def execution():
    execution_instance = {}

    if 'config_uuid' in request.form:
        config_uuid = request.form['config_uuid']
        config_entry = persistence['configuration'][config_uuid]
        execution_instance['config'] = config_entry

        # Assign values from config if they are stored in the config, otherwise assign None
        use_https = bool(config_entry['use_https']) if 'use_https' in config_entry else None
        method = str(config_entry['method']).upper() if 'method' in config_entry else None
        hostname = str(config_entry['hostname']) if 'hostname' in config_entry else None
        port = int(config_entry['port']) if 'port' in config_entry else None
        path = str(config_entry['path']) if 'path' in config_entry else None
        test_body = config_entry['test_body'] if 'test_body' in config_entry else None
        test_header = config_entry['test_header'] if 'test_header' in config_entry else None

        # Check if required parameters are set
        if use_https is not None and method and hostname and port and path:

            protocol = 'http'
            if use_https:
                protocol += 's'

            target_url = f'{protocol}://{hostname}:{port}{path}'

            # Send request with given parameters
            response = requests.request(method, target_url, headers=test_header, json=test_body)

            response_status = response.status_code

            # Create UUID for execution
            execution_uuid = str(uuid.uuid4())
            execution_instance['uuid'] = execution_uuid
            execution_instance['target_url'] = target_url
            execution_instance['status'] = str(response_status)

            persistence['execution'][execution_uuid] = execution_instance

            execution_results_dir = os.path.join(storage_path, execution_uuid)
            os.makedirs(execution_results_dir)

            execution_json = os.path.join(execution_results_dir, 'execution.json')
            received_response = os.path.join(execution_results_dir, 'response.bin')
            with open(execution_json, 'w') as exec_json:
                exec_json.write(json.dumps(execution_instance))

            with open(received_response, 'wb') as response_bin:
                response_bin.write(pickle.dumps(response))

            with tempfile.NamedTemporaryFile() as tf:
                tmp_zip_file = shutil.make_archive(tf.name, 'zip', execution_results_dir)
                shutil.copy2(tmp_zip_file, os.path.join(execution_results_dir, result_zip_file_name))

            # Test was executed with any possible outcome
            return jsonify(execution_instance), 200

        else:
            return "Required configuration parameters are missing.", jsonify(config_entry), 400
    else:
        return "No configuration with that ID found.", jsonify(persistence), 404


# Get execution results
@plugin.route('/execution/<string:exec_uuid>/', methods=['GET'])
def execution_results(exec_uuid):
    try:
        execution_uuid = persistence.get('execution').get(exec_uuid).get('uuid')
    except AttributeError:
        return "No execution found with that ID.", 404

    results_zip_path = os.path.join(storage_path, execution_uuid, result_zip_file_name)
    if os.path.isfile(results_zip_path):
        return send_file(results_zip_path)
    else:
        return "No results available (yet).", 404
