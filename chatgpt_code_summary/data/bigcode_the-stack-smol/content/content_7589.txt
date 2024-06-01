from flask import Flask, request, jsonify
from flask_jsonrpc import JSONRPC
# import json_to_db
import psycopg2
import sys
from obs import *
import config 

app = Flask(__name__)
app.config.from_object(config.DevelopmentMaxConfig)
jsonrpc = JSONRPC(app,'/api')


sys.path.insert(0,app.config['SQL_PATH'])
from sql_methods import *

sys.path.insert(0,app.config['SCRIPTS_PATH'])
from file_utils import insert_history_to_file


		
@app.route('/')
def index():
    return "Template to recieve data"

@app.route('/api/get_history', methods=['GET', 'POST'])
def get_history():
	content = request.get_json(force=True)
	insert_history_to_file(content, HISTORY_PATH) 
	return jsonify(content)

@app.route('/api/get_content', methods=['GET', 'POST'])
def get_content():
	content = ("""{}""".format(request.get_json(force=True))).replace('\'','\"')

	if content != "[]" and content:
		if content[0] != '[':
			content = '[' + content + ']'
		content += '\n\n'
		# print(content)
		client.put(content)
	return jsonify(content)

if __name__ == '__main__':
	client = Client("127.0.0.1", 8181, app.config['DB'], app.config['USER'], app.config['PASSWORD'], app.config['HOST'], app.config['PORT'])
	app.run(host='127.0.0.1', port= 5000)
	# json_insert.to_csv('/Users/MaximZubkov/Desktop/Programming/Python/Python_Project/analysis/son.csv')
	client.close()