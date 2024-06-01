import os
import json
import time
import datetime
import manageMonitoredUsersDB

pathToJSON = os.getcwd() + '/generatedJSON'

def get_local_json_timestamp_epoch(username, filename):
    monitoredJSON = None
    try:
        monitoredJSON = json.load(open(pathToJSON + os.sep + filename, "r+"))
    except:
        with open(os.getcwd() + '/logs/fail_to_get_local_epoch', "a") as fileText:
            fileText.write("The JSON fail to read is " + pathToJSON + os.sep + filename + " at " + str(datetime.datetime.now()) + "\n")
        fileText.close()
    if monitoredJSON == None:
        return None
    user_info = monitoredJSON["user_info"]
    json_timestamp_epoch = user_info["json_timestamp_epoch"]
    json_timestamp_epoch = float(json_timestamp_epoch)  #Epoch LOCAL
    return json_timestamp_epoch

def get_remote_json_timestamp_epoch(username):
    user_infoRemote = None
    monitoredUserSelected = manageMonitoredUsersDB.get_monitoredUserByName(username)
    temp = monitoredUserSelected[2]
    temp = temp.replace("'", "\"")
    temp = temp.replace("True", "true")
    temp = temp.replace("False", "false")
    temp = json.loads(temp)
    for key in temp.keys():
        if  key == "user_info":
            user_infoRemote = temp[key]
    if user_infoRemote != None:
        json_timestamp_epochRemote = user_infoRemote["json_timestamp_epoch"]
        return float(json_timestamp_epochRemote)  #Epoch REMOTO, el guardado en monitoredUser.db
    else: 
        print("\n" + "\033[91m" + "ERROR: No se ha podido obtener user_info en remoto, monitoredUser.db" + "\033[0m" +  "\n")
        with open(os.getcwd() + '/logs/fail_to_get_remote_epoch', "a") as fileText:
            fileText.write("The username fail to read is " + username  + " at " + str(datetime.datetime.now()) + "\n")
        fileText.close()


def checkArrivedJSON():
    for filename in sorted(os.listdir(pathToJSON)):
        if filename.endswith(".json"): 
            username = filename.strip(".json")
            #Obtención del epoch del JSON local
            json_timestamp_epoch = get_local_json_timestamp_epoch(username, filename)
            if json_timestamp_epoch == None:
                continue
            #Obtención del epoch del JSON remoto, en monitoredUser.db
            json_timestamp_epochRemote = get_remote_json_timestamp_epoch(username)
            #Comprobación del tiempo transcurrido entre local y remoto
            #print("\033[92m" + "json_timestamp_epoch: " + str(json_timestamp_epoch) + "\033[0m" +  "\n")
            #print("\033[92m" + "json_timestamp_epochRemote: " + str(json_timestamp_epochRemote) + "\033[0m" +  "\n")
            if json_timestamp_epoch > json_timestamp_epochRemote:
                monitoredJSON = json.load(open(pathToJSON + os.sep + filename, "r+"))
                monitoredJSON = str(monitoredJSON)
                manageMonitoredUsersDB.update_monitoredUserByName(username, monitoredJSON)

#MAIN
veces = 0
while True:
    checkArrivedJSON()
    time.sleep(1)
    if veces >= 10:
        print("Checking new user activities...\n")
        veces = 0
    veces += 1