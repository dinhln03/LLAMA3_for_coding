def is_sha1(maybe_sha):
    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True

def validate(date_text):
    try:
        datetime.datetime.strptime(date_text, '%d-%m-%Y:%S-%M-%H')
        return True
    except ValueError:
        return False

from flask_cors import CORS
from flask import Flask, render_template, Response, request, jsonify
import pandas as pd
import os
import json
import shutil
import datetime
import base64
import binascii
import datetime
import requests as r

LOGIN_FILE_NAME = "login.csv"
DB = "templates/images" 
GLOBAL_LIST = "acts.csv"
IP = "3.208.6.174:80"
INSTANCE_IP = "34.226.230.93"
count_requests = 0
#IP = "127.0.0.1:5000"


app = Flask(__name__)
CORS(app)

@app.errorhandler(405)
def method_not_allowed(e):
    global count_requests
    count_requests += 1
    return jsonify({'error': 405}), 405

@app.route("/")
def index():
    return render_template('index.html')
    
@app.route("/api/v1/categories", methods = ["GET", "POST"])
def list_categories():
    global count_requests
    count_requests += 1
    if not os.path.exists(DB):
        os.makedirs(DB, exist_ok = True)

    if request.method == 'GET':
        categories = os.listdir(DB)
        if not categories:
            return Response('{}', status=204, mimetype='application/json')
        response_data = {}
        
        for category in categories:
            response_data[category] = len(os.listdir(DB + "/" + category))        
        return jsonify(response_data)


    elif request.method == "POST":
        category = json.loads(request.data)[0]

        if category in os.listdir(DB):
            return Response('{}', status=400, mimetype='application/json')

        os.makedirs(DB + "/" + category, exist_ok = True)
        return Response('{}', status=201, mimetype='application/json')
        
    else:
        return Response('{}', status=405, mimetype='application/json')




@app.route("/api/v1/categories/<category>", methods = ["DELETE"])
def delete_category(category = None):
    global count_requests
    count_requests += 1
    if request.method == 'DELETE':
        categories = os.listdir(DB)
        if category in categories:
            if GLOBAL_LIST in os.listdir():
                data = pd.read_csv(GLOBAL_LIST)
                data = data[data.category != category]
                data.to_csv(GLOBAL_LIST, index = False)
            shutil.rmtree(DB + "/" + category)
            return Response('{}', status=200, mimetype='application/json')
        else:
            return Response('{}', status=400, mimetype='application/json')
    else:
        return Response('{}', status=405, mimetype='application/json')

    
@app.route("/api/v1/categories/<category>/acts", methods = ["GET"])
def list_acts(category = None):
    global count_requests
    count_requests += 1
    if request.method == 'GET':
        temp_path = DB + "/" + category + "/" + GLOBAL_LIST
        if category not in os.listdir(DB):
            return Response('[]', status=400, mimetype='application/json')
        start = request.args.get('start')
        end = request.args.get("end")
        if start == None and end == None:
            #print("This part")
            if os.path.exists(temp_path):
                data = pd.read_csv(temp_path)
                rows = data.shape[0]
                if rows == 0:
                    return Response('[]', status=204, mimetype='application/json')
                elif rows >= 100:
                    return Response('[]', status=413, mimetype='application/json')
                else:
                    response_data = data.to_json(orient = "records")
                    return Response(response_data, status=200, mimetype='application/json')
            else:
                return Response('[]', status=204, mimetype='application/json')
        else:
            start = int(start)
            end = int(end)
            temp_path = DB + "/" + category + "/" + GLOBAL_LIST
            if category not in os.listdir(DB):
                return Response('[]', status=400, mimetype='application/json')
            if os.path.exists(temp_path):
                data = pd.read_csv(temp_path)
                data["timestamp"] = pd.to_datetime(data["timestamp"], format = '%d-%m-%Y:%S-%M-%H')
                data["actId"] = data["actId"].astype(int)
                sorted_data = data.sort_values(["timestamp", "actId"], ascending = [False, False], axis = 0)
                #print(data)
                #print(sorted_data)
                rows = data.shape[0]
                if start < 1 or end > rows:
                    return Response('[]', status=400, mimetype='application/json')
                if rows == 0:
                    return Response('[]', status=204, mimetype='application/json')
                else:
                    required_data = pd.DataFrame(sorted_data.iloc[start-1: end, :])
                    #print(required_data)
                    if required_data.shape[0] > 100:
                        return Response("[]", status=413, mimetype='application/json')
                    required_data["timestamp"] = pd.to_datetime(required_data["timestamp"], format = '%d-%m-%Y:%S-%M-%H')
                    required_data["timestamp"] = required_data["timestamp"].astype(str)
                    response_data = required_data.to_json(orient = "records")
                    return Response(response_data, status=200, mimetype='application/json')
            else:
                return Response('[]', status=204, mimetype='application/json')
    else:
        return Response('{}', status=405, mimetype='application/json')

    
@app.route("/api/v1/categories/<category>/acts/size", methods = ["GET"])
def count_acts(category = None):
    global count_requests
    count_requests += 1
    if request.method == 'GET':
        temp_path = DB + "/" + category
        if category not in os.listdir(DB):
            return Response('[]', status=400, mimetype='application/json')
        if os.path.exists(temp_path):
            data = pd.read_csv(GLOBAL_LIST)
            count = data[data.category == category].shape[0]
            return Response('[{0}]'.format(str(count)), status=200, mimetype='application/json')
        else:
            return Response('[]', status=204, mimetype='application/json')
    else:
        return Response('{}', status=405, mimetype='application/json')

@app.route("/api/v1/acts/upvote", methods = ["POST"])
def upvote():
    global count_requests
    count_requests += 1
    if request.method == 'POST':
        act_id = int(json.loads(request.data)[0])
        data_id = pd.read_csv(GLOBAL_LIST)
        if act_id not in data_id["act_id"].tolist():
            return Response('[]', status=400, mimetype='application/json')

        category = data_id[data_id["act_id"] == act_id]["category"].iloc[0]
        temp_path = DB + "/" + category + "/" + GLOBAL_LIST

        data = pd.read_csv(temp_path)
        data.set_index("actId", inplace = True)
        data.at[act_id, "upvotes"] += 1
        data.reset_index(inplace = True)
        data.to_csv(temp_path,index = False)
        
        return Response("{}", status=200, mimetype='application/json')
    
    else:
        return Response('{}', status=405, mimetype='application/json')

@app.route("/api/v1/acts/<actId>", methods = ["DELETE"])
def delete_act(actId = None):
    global count_requests
    count_requests += 1
    if request.method == 'DELETE':
        act_id = int(actId)
        data_id = pd.read_csv(GLOBAL_LIST)
        
        if act_id not in data_id["act_id"].tolist():
            return Response('[]', status=400, mimetype='application/json')

        category = data_id[data_id["act_id"] == act_id]["category"].iloc[0]
        temp_path = DB + "/" + category + "/" + GLOBAL_LIST

        data_id = data_id[data_id["act_id"] != act_id]
        data_id.to_csv(GLOBAL_LIST, index = False)
        
        data = pd.read_csv(temp_path)
        data = data[data["actId"] != act_id]
        data.to_csv(temp_path, index = False)

        os.remove(DB + "/" + category + "/" + str(act_id) + ".png")
        return Response("{}", status=200, mimetype='application/json')
    
    else:
        return Response('{}', status=405, mimetype='application/json')


    
# @app.route("/api/v1/categories/<category>/acts?start=<startrange>&end=<endrange>", methods = ["GET"])
# def range_acts(category = None, startrange = 0, endrange = 0):
#     if request.method == 'GET':
#         temp_path = DB + "/" + category + "/" + GLOBAL_LIST
#         if category not in os.listdir(DB):
#             return Response('[]', status=400, mimetype='application/json')
#         if os.path.exists(temp_path):
#             data = pd.read_csv(temp_path)
#             sorted_data = data.sort(columns = ["timestamp"], ascending = False)
#             rows = data.shape[0]
#             if startrange < 1 or endrange > rows:
#                 return Response('[]', status=400, mimetype='application/json')
#             if rows == 0:
#                 return Response('[]', status=204, mimetype='application/json')
#             else:
#                 required_data = sorted_data.ix[startrange-1: endrange-1, :]
#                 print(required_data)
#                 if required_data.shape[0] > 100:
#                     return Response("[]", status=413, mimetype='application/json')
#                 response_data = required_data.to_json(orient = "records")
#                 return Response(response_data, status=200, mimetype='application/json')
#         else:
#             return Response('[]', status=204, mimetype='application/json')
#     else:
#         return Response('{}', status=405, mimetype='application/json')


@app.route("/api/v1/acts", methods = ["POST"])
def upload_act():
    global count_requests
    count_requests += 1
    if request.method == 'POST':
        if not os.path.exists(DB):
            os.makedirs(DB, exist_ok = True)
            
        request_data = json.loads(request.data.decode('utf-8'))

        if not GLOBAL_LIST in os.listdir():
            data = pd.DataFrame(columns = ['act_id', "category"])
            data.to_csv(GLOBAL_LIST, index = False)
            
        if not LOGIN_FILE_NAME in os.listdir():
            data = pd.DataFrame(columns = ['username', 'password'])
            data.to_csv(LOGIN_FILE_NAME, index = False)
            
        data_acts = pd.read_csv(GLOBAL_LIST)
        #data_users = pd.read_csv(LOGIN_FILE_NAME)
        # Username and actId
        header = {"origin": INSTANCE_IP}
        resp = r.get( "http://"+ IP + "/api/v1/users", "{}", headers = header)
        print("=============")
        print(resp.text)
        print("=============")
        data_users = eval(resp.text)
        if request_data['username'] not in data_users or request_data["actId"] in data_acts["act_id"].tolist():
            return Response('{}', status=400, mimetype='application/json')
        # Upvotes field
        if "upvotes" in request_data.keys():
            return Response('{}', status=400, mimetype='application/json')
        request_data['upvotes'] = 0
        # category name
        if request_data["categoryName"] not in os.listdir(DB):
            return Response('{}', status=400, mimetype='application/json')
        # Date Validity
        if not validate(request_data["timestamp"]):
            return Response('{}', status=400, mimetype='application/json')
        # Base64 validity
        try:
            base64.b64decode(request_data["imgB64"])
        except binascii.Error:
            return Response('{}', status=400, mimetype='application/json')

        data_acts = data_acts.append({"act_id": int(request_data["actId"]), "category": request_data["categoryName"] }, ignore_index = True)
        data_acts.to_csv(GLOBAL_LIST, index = False)

        with open(DB + "/" + request_data["categoryName"] + "/" +str(request_data["actId"]) + ".png", "wb") as fp:
            fp.write(base64.decodebytes(request_data["imgB64"].encode()))

        temp_path = DB + "/" + request_data["categoryName"] + "/" + GLOBAL_LIST
        if not GLOBAL_LIST in os.listdir(DB + "/" + request_data["categoryName"]):
            data = pd.DataFrame(columns = list(request_data.keys()))
            data.to_csv(temp_path, index = False)

        data = pd.read_csv(temp_path)
        data = data.append(request_data, ignore_index = True)
        data.to_csv(temp_path, index = False)

        return Response('{}', status=201, mimetype='application/json')
    else:
        return Response('{}', status=405, mimetype='application/json')


@app.route("/api/v1/acts/count", methods = ["GET"])
def count_act():
    global count_requests
    count_requests += 1
    if request.method == 'GET':
        if not GLOBAL_LIST in os.listdir():
            return Response('[0]', status=200, mimetype='application/json')
        else:
            data_acts = pd.read_csv(GLOBAL_LIST)
            count_acts = data_acts.shape[0]
            return Response('['+ str(count_acts) +']', status=200, mimetype='application/json')
    else:
        return Response('{}', status=405, mimetype='application/json')

@app.route("/api/v1/_count", methods = ["GET", "DELETE"])
def count_request():
    global count_requests
    if request.method == 'GET':
        return Response('['+ str(count_requests) +']', status=200, mimetype='application/json')
    elif request.method == 'DELETE':
        count_requests = 0
        return Response('{}', status=200, mimetype='application/json')
    else:
        return Response('{}', status=405, mimetype='application/json')

    
if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 80, threaded=True)
    #app.run(threaded = True, debug = True, port = 2000)
