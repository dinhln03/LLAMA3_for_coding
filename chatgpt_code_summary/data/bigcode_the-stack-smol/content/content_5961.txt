import boto3

from queuing_hub.conn.base import BasePub, BaseSub


class AwsBase():

    def __init__(self, profile_name=None):
        session = boto3.Session(profile_name=profile_name)
        self._client = session.client('sqs')
        self._queue_list = self._client.list_queues()['QueueUrls']


class AwsPub(AwsBase, BasePub):

    def __init__(self, profile_name=None):
        AwsBase.__init__(self, profile_name=profile_name)
        BasePub.__init__(self)

    @property
    def topic_list(self) -> list:
        return self._queue_list

    def push(self, topic: str, body: str) -> dict:
        response = self._client.send_message(
            QueueUrl=topic,
            MessageBody=body
        )
        return response['MessageId']


class AwsSub(AwsBase, BaseSub):

    ATTRIBUTE_NAMES = [
        'ApproximateNumberOfMessages',
        # 'ApproximateNumberOfMessagesDelayed',
        # 'ApproximateNumberOfMessagesNotVisible',
        # 'DelaySeconds',
        # 'MessageRetentionPeriod',
        # 'ReceiveMessageWaitTimeSeconds',
        # 'VisibilityTimeout'
    ]

    def __init__(self, profile_name=None):
        AwsBase.__init__(self, profile_name=profile_name)
        BaseSub.__init__(self)

    @property
    def sub_list(self) -> list:
        return self._queue_list

    def qsize(self, sub_list: list = None) -> dict:
        response = {'aws': {}}
        if not sub_list:
            sub_list = self._queue_list

        for sub in sub_list:
            response['aws'][sub] = self._get_message_count(sub)

        return response

    def is_empty(self, sub: str) -> bool:
        return self._get_message_count(sub) == 0

    def purge(self, sub: str) -> None:
        self._client.purge_queue(QueueUrl=sub)

    def pull(self, sub: str, max_num: int = 1, ack: bool = False) -> list:
        response = self._client.receive_message(
            QueueUrl=sub,
            MaxNumberOfMessages=max_num
        )

        messages = response.get('Messages')

        if ack and messages:
            self._ack(sub, messages)

        return [message.get('Body') for message in messages]

    def _ack(self, sub: str, messages: list) -> None:
        receipt_handle_list = \
            [message['ReceiptHandle'] for message in messages]
        for receipt_handle in receipt_handle_list:
            self._client.delete_message(
                QueueUrl=sub,
                ReceiptHandle=receipt_handle
            )

    def _get_message_count(self, sub: str) -> int:
        attributes = self._get_attributes(sub, self.ATTRIBUTE_NAMES)
        return int(attributes[self.ATTRIBUTE_NAMES[0]])

    def _get_attributes(self, sub: str, attribute_names: str) -> dict:
        response = self._client.get_queue_attributes(
            QueueUrl=sub,
            AttributeNames=attribute_names
        )
        return response['Attributes']
