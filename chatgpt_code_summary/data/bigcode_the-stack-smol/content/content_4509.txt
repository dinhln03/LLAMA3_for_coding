import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

import uuid

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['TESTING'] = True


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/message", methods=['POST'])
def receive_message():

    error_response = {"error": "No input_message in request"}

    if request.data:
        content = request.get_json()
        if "input_message" not in content:
            return error_response, 400

        input_message = content["input_message"]
        print(input_message)

        # TODO Pass the message through the model
        response_message = "Hello, " + input_message
        response = {"message": response_message}
        return response, 200

    # If anything goes wrong, return an error
    return error_response, 400


@app.route('/start', methods=['POST', 'GET'])  # TODO REMOVE GET
# @app.route('/start', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            ret = {'error': 'No selected file'}
            return ret, 415

        file = request.files['file']

        # If the user does not select a file, error
        if file.filename == '':
            ret = {'error': 'No selected file'}
            return ret, 415

        # Check if allowed and secure
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # Create a unique filename and save it locally
            if filename in os.listdir(app.config['UPLOAD_FOLDER']):
                filename = str(uuid.uuid4()) + '.' + \
                    filename.rsplit('.', 1)[-1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # TODO Train our model

            ret = {
                'message': 'file uploaded successfully',
                'filename': filename
            }

            return ret, 200

    # Temporary have an upload page for testing
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=submit value=Upload>
    </form>
    '''
