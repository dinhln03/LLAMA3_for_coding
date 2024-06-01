from kafka import KafkaProducer
from json import dumps as json_dumps, load as json_load
import time


class ProducerServer(KafkaProducer):

    def __init__(self, input_file, topic, **kwargs):
        super().__init__(**kwargs)
        self.input_file = input_file
        self.topic = topic

    def generate_data(self):
        with open(self.input_file) as f:
            data = json_load(f)
            for line in data:
                message = self.dict_to_binary(line)
                self.send(self.topic, message)

    def dict_to_binary(self, json_dict):
        return json_dumps(json_dict).encode('utf-8')