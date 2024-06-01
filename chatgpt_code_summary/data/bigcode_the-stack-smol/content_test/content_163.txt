import json
from time import sleep
from uuid import uuid4
from datetime import datetime
import logging
from kafka import KafkaProducer, KafkaConsumer
from settings import (
    KAFKA_BOOTSTRAP_SERVER,
    KAFKA_VALUE_ENCODING,
    KAFKA_INBOUND_TOPIC,
    KAFKA_SUCCESS_OUTBOUND_TOPIC,
    KAFKA_ERROR_OUTBOUND_TOPIC,
    KAFKA_DEAD_LETTER_QUEUE_TOPIC,
    KAFKA_SUCCESS_ACKS,
    KAFKA_ERROR_ACKS,
    KAFKA_DEAD_LETTER_QUEUE_ACKS,
    KAFKA_INBOUND_GROUP_ID,
    KAFKA_INBOUND_AUTO_OFFSET_RESET,
    EXECUTION_SLEEP,
    EXECUTION_MESSAGE_FORCE_ERROR_KEY,
    KAFKA_INBOUND_TIMEOUT,
    KAFKA_INBOUND_MAX_RECORDS,
)
from schemas import ResultField


LOGGER = logging.getLogger(__name__)


class RequestsProcessorBuilder(object):
    @staticmethod
    def build():
        return RequestsProcessor(
            RequestsProcessorBuilder.build_inbound_consumer(),
            RequestsProcessorBuilder.build_success_publisher(),
            RequestsProcessorBuilder.build_error_publisher(),
            RequestsProcessorBuilder.build_dead_letter_publisher(),
        )

    @staticmethod
    def build_inbound_consumer():
        return KafkaConsumer(
            KAFKA_INBOUND_TOPIC,
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVER],
            auto_offset_reset=KAFKA_INBOUND_AUTO_OFFSET_RESET,
            enable_auto_commit=False,
            group_id=KAFKA_INBOUND_GROUP_ID,
            value_deserializer=lambda value: json.loads(value.decode(KAFKA_VALUE_ENCODING))
        )

    @staticmethod
    def build_success_publisher():
        return RequestsProcessorBuilder.build_producer(KAFKA_SUCCESS_ACKS)

    @staticmethod
    def build_error_publisher():
        return RequestsProcessorBuilder.build_producer(KAFKA_ERROR_ACKS)

    @staticmethod
    def build_dead_letter_publisher():
        return RequestsProcessorBuilder.build_producer(KAFKA_DEAD_LETTER_QUEUE_ACKS)

    @staticmethod
    def build_producer(acknowledgements):
        return KafkaProducer(
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVER],
            value_serializer=lambda value: json.dumps(value).encode(KAFKA_VALUE_ENCODING),
            acks=acknowledgements
        )


class RequestsProcessor(object):

    def __init__(self, inbound_consumer, success_publisher, error_publisher, dead_letter_publisher):
        self.inbound_consumer = inbound_consumer
        self.success_publisher = success_publisher
        self.error_publisher = error_publisher
        self.dead_letter_publisher = dead_letter_publisher

    def start(self):
        while True:
            messages_by_partition = self.inbound_consumer.poll(
                timeout_ms=KAFKA_INBOUND_TIMEOUT,
                max_records=KAFKA_INBOUND_MAX_RECORDS,
            )
            self.handle_messages(messages_by_partition)

    def handle_messages(self, messages_by_partition):
        for topic_partition, messages in messages_by_partition.items():
            for message in messages:
                self.handle_message(topic_partition, message)

    def handle_message(self, topic_partition, message):
        execution = message.value

        LOGGER.info("Handling message: '%s'", str(execution))

        try:
            failed, outputs, start_time, end_time, total_seconds = RequestsProcessor.process(
                execution
            )

            result = RequestsProcessor.build_result(
                execution, outputs, start_time, end_time, total_seconds
            )

            self.publish_to_result_topic(result, failed)
        except:
            LOGGER.exception("An error occurred while handling the execution")
            self.publish_to_dead_letter_queue_topic(execution)

        self.commit_current_message(topic_partition)

        LOGGER.info("Done handling message: '%s'", str(execution))

    def publish_to_result_topic(self, execution, failed):
        if failed:
            LOGGER.info("Publishing execution to failed executions topic")
            self.error_publisher.send(KAFKA_ERROR_OUTBOUND_TOPIC, value=execution)
            LOGGER.info("Published execution to failed executions topic")
        else:
            LOGGER.info("Publishing execution to successful executions topic")
            self.success_publisher.send(KAFKA_SUCCESS_OUTBOUND_TOPIC, value=execution)
            LOGGER.info("Published execution to successful executions topic")

    def publish_to_dead_letter_queue_topic(self, execution):
        LOGGER.info("Publishing execution to dead letter queue topic")
        self.dead_letter_publisher.send(KAFKA_DEAD_LETTER_QUEUE_TOPIC, value=execution)
        LOGGER.info("Published execution to dead letter queue topic")

    def commit_current_message(self, topic_partition):
        LOGGER.info("Committing")
        self.inbound_consumer.commit()
        new_offset = self.inbound_consumer.committed(topic_partition)
        LOGGER.info("Committed. New Kafka offset: %s", new_offset)

    @staticmethod
    def process(execution):
        LOGGER.info("Executing: %s", execution)

        start_time = datetime.utcnow()

        failed, outputs = Executor(execution).execute()

        end_time = datetime.utcnow()
        processing_time_difference = end_time - start_time
        processing_time_seconds = processing_time_difference.total_seconds()

        LOGGER.info("Executed: %s", execution)

        return failed, outputs, start_time, end_time, processing_time_seconds

    @staticmethod
    def build_result(execution, outputs, start_time, end_time, total_seconds):
        return {
            ResultField.ID: generate_identifier(),
            ResultField.START_TIME: str(start_time),
            ResultField.END_TIME: str(end_time),
            ResultField.TOTAL_SECONDS: total_seconds,
            ResultField.EXECUTION: execution.copy(),
            ResultField.OUTPUTS: outputs
        }


class Executor(object):

    def __init__(self, execution):
        self.execution = execution

    def execute(self):
        Executor.wait(EXECUTION_SLEEP)
        force_error = self.execution.get(EXECUTION_MESSAGE_FORCE_ERROR_KEY)
        outputs = Executor.get_outputs(force_error)
        return force_error, outputs

    @staticmethod
    def wait(seconds):
        LOGGER.info("Sleeping for %d seconds...", seconds)
        sleep(seconds)
        LOGGER.info("Done waiting")

    @staticmethod
    def get_outputs(force_error):
        outputs = {}
        if not force_error:
            outputs[ResultField.OUTPUT_MESSAGE_KEY] = ResultField.OUTPUT_MESSAGE_VALUE
        return outputs


def generate_identifier():
    return str(uuid4())
