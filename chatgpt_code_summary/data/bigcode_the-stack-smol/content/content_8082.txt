import paho.mqtt.client as mqtt
import time
import argparse
from tinydb import TinyDB, Query
from tinyrecord import transaction
import logging
import sys
import json
import threading
import ssl
from random import randint

CA_ROOT_CERT_FILE = "ag-certificate/AmazonRootCA1.pem"
THING_CERT_FILE = "ag-certificate/..."
THING_PRIVATE_KEY = "ag-certificate/..."

# init args parser
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "MQTT_broker", metavar="MQTT_broker", type=str, help="Address of the MQTT broker"
)
args = parser.parse_args()

# init logger
logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
)
logger = logging.getLogger()
fileHandler = logging.FileHandler("{0}/{1}.log".format("log", f"agenceur"))
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.INFO)

# init opaque DB
db_opaque = TinyDB("opaque.json")

# init clear measures DB
db_measures = TinyDB("measures.json")
db_measures.truncate()
lock = threading.Lock()  # on received message


def on_message(client, userdata, message):
    with lock:
        logger.debug(
            "rcvd: " + message.topic + "/" + str(message.payload.decode("utf-8"))
        )

        if message.topic == "addToPool":
            # store in DB
            logger.info(f"storing payload")
            db_opaque.insert({"entry": str(message.payload.decode("utf-8"))})

        if message.topic == "requestPool":
            asking = str(message.payload.decode("utf-8"))
            logger.info(f"received pool request from {asking}")
            completePool = db_opaque.all()
            # truncate because we will save it again?
            db_opaque.truncate()
            # dont sent if pool is empty
            if len(completePool):
                to_send = []
                if len(completePool) > 10:
                    for i in range(10):
                        to_send.append(completePool.pop(0))
                else:
                    to_send = completePool.copy()
                    completePool.clear()
                
                for left in completePool:
                    db_opaque.insert(left)

                #logger.info(f"to_send: {to_send}")
                table_json = json.dumps(to_send)
                logger.info(f"publishing table to getPool{asking}, len={len(table_json)}, n={len(to_send)}")
                client.publish(f"getPool{asking}", table_json, qos=1)

        if message.topic == "measures":
            j = json.loads(message.payload.decode("utf-8"))
            logger.info(f"m: {message.payload}")
            db_measures.insert({"entry": message.payload.decode("utf-8")})
            logger.info(f"received measure {j['MUID']}")


# connecting to MQTT broker
logger.info(f"Connecting to broker at {args.MQTT_broker}")
client = mqtt.Client("Agenceur")


client.tls_set(CA_ROOT_CERT_FILE, certfile=THING_CERT_FILE, keyfile=THING_PRIVATE_KEY)#, cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2, ciphers=None)
client.connect(args.MQTT_broker, 8883)
# client.enable_logger()

# start receive thread
client.loop_start()

# subscribe to
# * addToPool:      endpoint for opaque payload
# * requestPool:    endpoint for opaque pool request from devices
# * measures:       endpoint for clear-data measures
client.subscribe("addToPool")
client.subscribe("requestPool")
client.subscribe("measures")

# register receive routine
client.on_message = on_message

# only on event execution
while True:
    time.sleep(1)

client.loop_stop()
