import requests
import argparse
import logging
import coloredlogs
import threading
from flask import Flask, request, jsonify
from flask_swagger import swagger
from waitress import serve
import subprocess
import json
from kafka import KafkaConsumer
from threading import Thread
from threading import Timer
from datetime import timedelta
import psycopg2
import time

app = Flask(__name__)
logger = logging.getLogger("DCSRestClient")
signalling_metric_infrastructure = {'expId': 'internal', 'topic': 'signalling.metric.infrastructure'}
signalling_metric_application = {'expId': 'internal', 'topic': 'signalling.metric.application'}
signalling_kpi = {'expId': 'internal', 'topic': 'signalling.kpi'}
dcm_port = "8090"
dcm_subscribe_url = "/dcm/subscribe"
dcm_unsubscribe_url = "/dcm/unsubscribe"
dcs_dashboard_url = "http://127.0.0.1:8080/portal/dcs/dashboard"
signalling_start = False

@app.route('/', methods=['GET'])
def server_status():
    """
    Get status.
    ---
    describe: get status
    responses:
      200:
        description: OK
    """
    logger.info("GET /")
    return '', 200

@app.route("/spec", methods=['GET'])
def spec():
    """
    Get swagger specification.
    ---
    describe: get swagger specification
    responses:
      swagger:
        description: swagger specification
    """
    swag = swagger(app)
    swag['info']['version'] = "1.0"
    swag['info']['title'] = "DCS REST API"
    return jsonify(swag)

def kafka_consumer_refresh_dashboard_handler(topic, value):
    logger.info("Creating Kafka Consumer for %s topic", topic)
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=[dcm_ip_address + ":9092"],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id=None,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')))

    message_received = False
    while not message_received:
        message = consumer.poll(timeout_ms=1000)
        if message != {}:
            logger.info("Message received in %s topic: %s", topic, message)
            message_received = True

    time.sleep(5)
    logger.info("Creating dashboard for topic: %s", topic)            
    r = requests.post(dcs_dashboard_url, json={'records': [ { 'value': json.loads(value) }]})
    logger.info("Response: Code %s", r)

    # This call seems that is not needed as the dashboard is generated when data is present.
    #time.sleep(2)            
    #logger.info("Refreshing dashboard for %s topic", topic)
    #subprocess.call(['/bin/bash', '/usr/bin/dcs/refresh_dashboard.sh', topic])

    logger.info("Closing Kafka Consumer for %s topic", topic)
    consumer.close()

def index_cleaner(topic, value):

    logger.info("Time to delete the dashboard for topic %s", topic)
    r = requests.delete(dcs_dashboard_url, json={'records': [ { 'value': json.loads(value) }]})
    logger.info("Response: Code %s", r)

    logger.info("Time to delete the Elasticsearch index for topic %s", topic)
    subprocess.call(['/bin/bash', '/usr/bin/dcs/delete_logstash_pipeline.sh', topic, 'yes'])

def kafka_consumer_signalling_topic_handler(signalling_topic_data):
    logger.info("Creating Kafka Consumer for %s topic", signalling_topic_data["topic"])
    consumer = KafkaConsumer(
        signalling_topic_data["topic"],
        bootstrap_servers=[dcm_ip_address + ":9092"],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id=None,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')))

    while signalling_start:
        message = consumer.poll(timeout_ms=1000)
        if message != {}:
            logger.info("Message received in %s topic: %s", signalling_topic_data["topic"], message)
            for tp, messages in message.items():
                for msg in messages:
                    logger.info("Value: %s", msg.value)
                    topic = json.loads(msg.value)["topic"]
                    if json.loads(msg.value)["action"] == "subscribe":
                        logger.info("Create Logstash pipeline for topic %s", topic)
                        subprocess.call(['/bin/bash', '/usr/bin/dcs/create_logstash_pipeline.sh', topic])

                        # Dashboard creation is commented because it will be created when data is published in the topic.
                        #r = requests.post(dcs_dashboard_url, json={'records': [ { 'value': json.loads(msg.value) }]})
                        #logger.info("Response: Code %s", r)

                        # Create Kafka consumer to wait for the first message received in the topic and, then, refresh the dashboard.
                        thread = threading.Thread(target = kafka_consumer_refresh_dashboard_handler, args = [topic, msg.value])
                        thread.start()

                        # Finally, save topic in DB
                        try:
                            connection = psycopg2.connect(user = "eve", password = eve_db_password, host = "localhost", port = "5432", dbname="pipelines")
                            logger.info("Inserting %s topic in database", topic)
                            cursor = connection.cursor()
                            cursor.execute("INSERT INTO pipeline VALUES ( %s )", (topic,))
                            connection.commit()
                            logger.info("Topic %s inserted in database", topic)
                            cursor.close()
                            connection.close()
                        except (Exception, psycopg2.Error) as error:
                            logger.error("Error while connecting to PostgreSQL: ", error)

                    elif json.loads(msg.value)["action"] == "unsubscribe":
                        logger.info("Delete Logstash pipeline for topic %s", topic)
                        subprocess.call(['/bin/bash', '/usr/bin/dcs/delete_logstash_pipeline.sh', topic, 'no'])
                        
                        # Schedule the removal of Kibana dashboard and Elasticsearch index (retention time of 14 days)
                        scheduled_thread = threading.Timer(timedelta(days=14).total_seconds(), index_cleaner, args = [topic, msg.value])
                        # This call is for testing purposes, to be commented when unused:
                        #scheduled_thread = threading.Timer(timedelta(seconds=30).total_seconds(), index_cleaner, args = [topic, msg.value])
                        scheduled_thread.start()
                        logger.info("Data removal for topic %s scheduled in 14 days", topic)

                        # Finally, delete topic in DB
                        try:
                            connection = psycopg2.connect(user = "eve", password = eve_db_password, host = "localhost", port = "5432", dbname="pipelines")
                            logger.info("Deleting %s topic in database", topic)
                            cursor = connection.cursor()
                            cursor.execute("DELETE FROM pipeline WHERE topic = %s", (topic,))
                            connection.commit()
                            logger.info("Topic %s deleted in database", topic)
                            cursor.close()
                            connection.close()
                        except (Exception, psycopg2.Error) as error:
                            logger.error("Error while connecting to PostgreSQL: ", error)
                    else:
                        logger.error("Action not allowed")

    logger.info("Closing Kafka Consumer for %s topic", signalling_topic_data["topic"])
    consumer.close()

def start_consuming_signalling_topic(signalling_topic_data):
    signalling_topic_data = json.loads(signalling_topic_data)
    logger.info("Starting %s topic", signalling_topic_data["topic"])
    logger.info("Sending POST request to %s", url_subscribe)
    # Send the request to the DCM.
    r = requests.post(url_subscribe, json=signalling_topic_data)
    logger.info("Response: Code %s", r)

    # Create Kafka consumer.
    global signalling_start
    signalling_start = True
    thread = threading.Thread(target = kafka_consumer_signalling_topic_handler, args = [signalling_topic_data])
    thread.start()

@app.route('/portal/dcs/start_signalling/', methods=['POST'])
def start_dcs():
    """
    Start signalling topics.
    ---
    describe: start signalling topics
    responses:
      201:
        description: accepted request
      400:
        description: error processing the request
    """
    logger.info("Request received - POST /portal/dcs/start_signalling/")
    try:
        start_consuming_signalling_topic(json.dumps(signalling_metric_infrastructure))
        start_consuming_signalling_topic(json.dumps(signalling_metric_application))
        start_consuming_signalling_topic(json.dumps(signalling_kpi))
    except Exception as e:
        logger.error("Error while parsing request")
        logger.exception(e)
        return str(e), 400
    return '', 201

def stop_consuming_signalling_topic(signalling_topic_data):
    signalling_topic_data = json.loads(signalling_topic_data)
    logger.info("Stopping %s topic", signalling_topic_data["topic"])
    logger.info("Sending DELETE request to %s", url_unsubscribe)
    # Send the request to the DCM.
    r = requests.delete(url_unsubscribe, json=signalling_topic_data)
    logger.info("Response: Code %s", r)

    # Delete Kafka consumer.
    global signalling_start
    # Put signalling_start to False, and then threads will finish their execution.
    signalling_start = False

@app.route('/portal/dcs/stop_signalling/', methods=['DELETE'])
def stop_dcs():
    """
    Stop signalling topics.
    ---
    describe: stop signalling topics
    responses:
      201:
        description: accepted request
      400:
        description: error processing the request
    """
    logger.info("Request received - DELETE /portal/dcs/stop_signalling/")
    try:
        stop_consuming_signalling_topic(json.dumps(signalling_metric_infrastructure))
        stop_consuming_signalling_topic(json.dumps(signalling_metric_application))
        stop_consuming_signalling_topic(json.dumps(signalling_kpi))
    except Exception as e:
        logger.error("Error while parsing request")
        logger.exception(e)
        return str(e), 400
    return '', 201

def checkValidPort(value):
    ivalue = int(value)
    # RFC 793
    if ivalue < 0 or ivalue > 65535:
        raise argparse.ArgumentTypeError("%s is not a valid port" % value)
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dcm_ip_address",
        help='DCM IP address, default IP is localhost',
        default='localhost')
    parser.add_argument(
        "--eve_db_password",
        help='DB password for eve user')
    parser.add_argument(
        "--port",
        type=checkValidPort,
        help='The port you want to use as an endpoint, default port is 8091',
        default="8091")
    parser.add_argument(
        "--log",
        help='Sets the Log Level output, default level is "info"',
        choices=[
            "info",
            "debug",
            "error",
            "warning"],
        nargs='?',
        default='info')
    args = parser.parse_args()
    numeric_level = getattr(logging, str(args.log).upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    coloredlogs.install(
        fmt='%(asctime)s %(levelname)s %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S',
        level=numeric_level)
    logging.basicConfig(filename='/var/log/dcs_rest_client.log')
    logging.getLogger("DCSRestClient").setLevel(numeric_level)
    logging.getLogger("requests.packages.urllib3").setLevel(logging.ERROR)
    args = parser.parse_args()
    logger.info("Serving DCSRestClient on port %s", str(args.port))
    global dcm_ip_address 
    dcm_ip_address= str(args.dcm_ip_address)
    global url_subscribe
    url_subscribe = "http://" + dcm_ip_address + ":" + dcm_port + dcm_subscribe_url
    global url_unsubscribe
    url_unsubscribe = "http://" + dcm_ip_address + ":" + dcm_port + dcm_unsubscribe_url
    global eve_db_password
    eve_db_password= str(args.eve_db_password)
    #TODO: advanced feature - connect to the database and make sure that Logstash pipelines are created for the topics saved in the DB.
    serve(app, host='0.0.0.0', port=args.port)
