#!/usr/bin/env python

import paho.mqtt.client as mqtt
import random

from logger import error, info
from message import Message
from pubsub import publish

MQTT_ERR_SUCCESS = 0

# MQTT client wrapper for use with Mainflux.
class MQTT:
    # Initialize the class with topics that will be used
    # during publish and possibly subscribe.
    def __init__(self, topics, client_id='mqtt-client', clean_session=True, qos=0, queue=None):
        info('mqtt', 'init')
        self.connected = False
        self.qos = qos
        self.queue = queue
        self.subscribe = queue != None

        # Handle topics string or slice.
        if isinstance(topics, basestring):
            topics = topics.split(',')
        self.topics = topics

        # Add randomness to client_id.
        client_id = client_id+'-'+str(random.randint(1000, 9999))

        self.client = mqtt.Client(client_id=client_id, clean_session=clean_session)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_publish = self.on_publish
        self.client.on_subscribe = self.on_subscribe

    # Connect to the MQTT adapter endpoint and start the internal
    # paho-mqtt loop.
    def connect(self, host, port, username, password):
        info('mqtt', 'connect')
        self.client.username_pw_set(username, password)
        self.client.connect(host, port=port, keepalive=60)
        self.client.loop_start()

    # Disconnect the client.
    def disconnect(self):
        self.connected = False
        self.client.loop_stop()
        self.client.disconnect()

    dur_count = 0.0
    dur_total = 0.0
    def dur_avg(self):
        return self.dur_total / self.dur_count

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        info('mqtt', 'on_connect '+str(rc))
        if rc == MQTT_ERR_SUCCESS:
            self.connected = True

        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        if self.subscribe:
            subs = []
            for topic in self.topics:
                info('mqtt', 'subscribe: channels/'+topic+'/messages, qos: '+str(self.qos))
                subs.append(('channels/'+topic+'/messages', self.qos))
            info('mqtt', 'subscriptions: '+str(subs))
            self.client.subscribe(subs)

    # When the client disconnects make sure to stop the loop.
    def on_disconnect(self, client, userdata, rc):
        info('mqtt', 'on_disconnect')
        if rc != MQTT_ERR_SUCCESS:
            info('mqtt', 'on_disconnect unexpected: '+str(rc))
            #self.disconnect()

        #self.client.reconnect()


    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        info('mqtt', 'on_message:'+msg.topic+': '+str(msg.payload))
        try:
            if self.queue:
                m = Message(msg.topic, msg.payload)
                if m.is_valid():
                    if m.for_device() and not m.get_name() in ['CONNECTED', 'REGISTERED', 'TX']:
                        self.queue.put(m)
                        publish(m, channel='inbound')
                    else:
                        publish(m, channel='inbound')
        except Exception as ex:
            error('mqtt', 'on_message: '+str(ex))

    # When a message has been published.
    def on_publish(self, client, userdata, mid):
        info('mqtt', 'on_publish: '+str(mid))

    # When a subscription is complete.
    def on_subscribe(client, userdata, mid, granted_qos):
        info('mqtt', 'on_subscribe mid: '+mid)

    # Publish a message to the topic provided on init.
    def publish(self, msg=None):
        if msg:
            info('mqtt', 'publish: '+str(msg))
            mid = self.client.publish(msg.topic, payload=msg.payload_str(), qos=self.qos)
            self.dur_count += 1
            self.dur_total += msg.get_duration()
            info('mqtt', 'published '+str(self.dur_count)+' for an avg. duration of '+str(self.dur_avg())+' secs. with '+str(self.dur_total)+' secs. in total')
