from kafka import KafkaConsumer

KAFKA_SERVER_URL = 'localhost:9092'
LOGIN = "bob"
PWD = "bob-secret"
TOPIC = "test-topic"
GROUP_ID = 'bob-group'

consumer = KafkaConsumer(TOPIC, group_id=GROUP_ID, bootstrap_servers=KAFKA_SERVER_URL,
                         security_protocol="SASL_PLAINTEXT",
                         sasl_mechanism='PLAIN', sasl_plain_username=LOGIN, sasl_plain_password=PWD)

for msg in consumer:
    print(msg)
