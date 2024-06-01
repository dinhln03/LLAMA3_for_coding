#from mq import *
import sys, time
import urllib3
#networking library
import json
try:
    print("Press CTRL+C to abort.")

    #mq = MQ();
    while True:
        http = urllib3.PoolManager()
        #perc = mq.MQPercentage()
        sys.stdout.write("\r")
        sys.stdout.write("\033[K")
        data = {
            "error":False,
            "device_id":"device123",
            "fuse_stat":["0","1","0","1","0"]
        }
        encoded_data = json.dumps(data).encode('utf-8')#create JSON object
        http.request(
            'POST',
            'http://192.168.43.156/smartdbbox/api/public/api/device/db/update',#IP add server
            body=encoded_data,
            headers={'Content-Type': 'application/json'} )
        sys.stdout.flush()
        time.sleep(0.1)



except:
    print("\nAbort by user")