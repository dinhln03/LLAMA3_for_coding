import time
from check_lang import check_py,check_rb,check_j,check_c,check_cpp
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import subprocess
import json
from json import JSONEncoder

from main import predict

app = Flask(__name__)
CORS(app)
@app.route("/")
def hello():
    return '<form action="/check" method="POST"><input name="code" size="135"><input type="submit" value="Code Here"></form>'

@app.route("/check", methods=['POST'])
def echo():
    codes = []
    filename = str(int(time.time()))
    dataDict = json.loads(request.data)



    # print dataDict
    # print "------------"

    with open('code/'+filename,'w+') as outfile:
        outfile.write(str(dataDict['sc']))



    codes.append(int(check_c("code/"+filename)))
    codes.append(int(check_cpp("code/"+filename)))
    codes.append(int(check_py("code/"+filename)))
    codes.append(int(check_rb("code/"+filename)))
    codes.append(1)

    print codes

    zero = 0
    count = 0
    correct_count = 0
    for code in codes:
        count = count+1
        if code==0:
            zero = zero + 1
            correct_count = count

    print zero

    if(zero == 1):
        if(correct_count==1):
            jsonString = {'cpp': 0.0, 'ruby': 0.0, 'c': 1.0, 'py': 0.0, 'java': 0.0}
            return jsonify(jsonString)
        elif(correct_count==2):
            jsonString = {'cpp': 1.0, 'ruby': 0.0, 'c': 0.0, 'py': 0.0, 'java': 0.0}
            return jsonify(jsonString)
        elif(correct_count==3):
            jsonString = {'cpp': 0.0, 'ruby': 0.0, 'c': 0.0, 'py': 1.0, 'java': 0.0}
            return jsonify(jsonString)
        elif(correct_count==4):
            jsonString = {'cpp': 0.0, 'ruby': 1.0, 'c': 0.0, 'py': 0.0, 'java': 0.0}
            return jsonify(jsonString)
    else:
        x = predict(dataDict['sc'])
        print x
        # return JSONEncoder().encode(x)
        return jsonify({'cpp': round(x['cpp'], 2), 'ruby': round(x['ruby'], 2), 'c': round(x['c'], 2), 'py': round(x['py'], 2), 'java': round(x['java'], 2)})
        #if score of cpp is eqgreater than 0.5 then run it to check if it runs then cpp else java
        # sa = []

        # score_cpp = x['cpp']
        # score_ruby = x['ruby']
        # score_c = x['c']
        # score_py = x['py']
        # score_java = x['java']
        #
        # sa.append(score_c)
        # sa.append(score_cpp)
        # sa.append(score_java)
        # sa.append(score_py)
        # sa.append(score_ruby)
        #
        # print sa


    # return ''.join([str(code) for code in codes])+" "+str(x)


if __name__ == "__main__":
    app.run(host= '0.0.0.0')
