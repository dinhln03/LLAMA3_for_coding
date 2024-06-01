from queue import Queue, Empty, Full

from ..core import DriverBase, format_msg
import pika


class Driver(DriverBase):
    def __init__(self, exchange, queue, routing_key=None, buffer_maxsize=None,
                 *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs
        self._exchange = exchange
        self._queue = queue
        self._routing_key = routing_key or queue
        self._buffer = Queue(buffer_maxsize) \
            if buffer_maxsize is not None else None
        self._declared = False

    def run(self, driver_id, ts, fields, tags):
        if not fields:
            return
        msg = format_msg(ts, driver_id, tags, fields)
        try:
            with pika.BlockingConnection(
                 pika.ConnectionParameters(*self._args, **self._kwargs)) as c:
                channel = c.channel()
                self._publish(channel, msg)
                # Flush buffer
                if self._buffer is not None:
                    try:
                        while True:
                            msg = self._buffer.get_nowait()
                            self._publish(channel, msg)
                    except Empty:
                        pass
        except pika.exceptions.AMQPError:
            # Add to buffer
            if self._buffer is not None:
                try:
                    self._buffer.put_nowait(msg)
                except Full:
                    pass

    def _declare(self, channel):
        if not self._declared:
            channel.exchange_declare(exchange=self._exchange, durable=True)
            channel.queue_declare(queue=self._queue, durable=True)
            channel.queue_bind(
                exchange=self._exchange,
                queue=self._queue,
                routing_key=self._routing_key
            )
            self._declared = True

    def _publish(self, channel, msg):
        self._declare(channel)
        channel.basic_publish(
            exchange=self._exchange,
            routing_key=self._routing_key,
            body=msg,
            properties=pika.BasicProperties(delivery_mode=2)
        )
