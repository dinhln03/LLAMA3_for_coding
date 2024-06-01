#!/usr/bin/env python

import logging
from typing import (
    Optional,
    Dict,
    List, Any)

from hummingbot.core.data_type.order_book import OrderBook
from sqlalchemy.engine import RowProxy

import hummingbot.connector.exchange.binarz.binarz_constants as constants
from hummingbot.connector.exchange.binarz.binarz_order_book_message import BinarzOrderBookMessage
from hummingbot.connector.exchange.binarz.binarz_websocket import BinarzTrade
from hummingbot.core.data_type.order_book_message import (
    OrderBookMessage, OrderBookMessageType
)
from hummingbot.logger import HummingbotLogger

_logger = None


class BinarzOrderMatched:
    def __init__(self):
        pass


class BinarzOrderBook(OrderBook):
    @classmethod
    def logger(cls) -> HummingbotLogger:
        global _logger
        if _logger is None:
            _logger = logging.getLogger(__name__)
        return _logger

    @classmethod
    def snapshot_message_from_exchange(cls,
                                       msg: Dict[str, Any],
                                       timestamp: float,
                                       *args, **kwargs):
        """
        Convert json snapshot data into standard OrderBookMessage format
        :param msg: json snapshot data from live web socket stream
        :param timestamp: timestamp attached to incoming data
        :return: BinarzOrderBookMessage
        """

        return BinarzOrderBookMessage(
            message_type=OrderBookMessageType.SNAPSHOT,
            content=msg,
            timestamp=timestamp,
            *args, **kwargs)

    @classmethod
    def snapshot_message_from_db(cls, record: RowProxy):
        """
        *used for backtesting
        Convert a row of snapshot data into standard OrderBookMessage format
        :param record: a row of snapshot data from the database
        :return: BinarzBookMessage
        """
        return BinarzOrderBookMessage(
            message_type=OrderBookMessageType.SNAPSHOT,
            content=record.json,
            timestamp=record.timestamp
        )

    @classmethod
    def diff_message_from_exchange(cls,
                                   msg: Dict[str, any],
                                   timestamp: Optional[float] = None):
        """
        Convert json diff data into standard OrderBookMessage format
        :param msg: json diff data from live web socket stream
        :param timestamp: timestamp attached to incoming data
        :return: BinarzOrderBookMessage
        """

        return BinarzOrderBookMessage(
            message_type=OrderBookMessageType.DIFF,
            content=msg,
            timestamp=timestamp
        )

    @classmethod
    def diff_message_from_db(cls, record: RowProxy):
        """
        *used for backtesting
        Convert a row of diff data into standard OrderBookMessage format
        :param record: a row of diff data from the database
        :return: BinarzBookMessage
        """
        return BinarzOrderBookMessage(
            message_type=OrderBookMessageType.DIFF,
            content=record.json,
            timestamp=record.timestamp
        )

    @classmethod
    def trade_message_from_exchange(cls,
                                    msg: BinarzTrade,
                                    timestamp: Optional[float] = None,
                                    ):
        """
        Convert a trade data into standard OrderBookMessage format
        """
        msg = {
            "exchange_order_id": msg.order_id,
            "trade_type": msg.type,
            "price": msg.price,
            "amount": msg.amount,
        }

        return BinarzOrderBookMessage(
            message_type=OrderBookMessageType.TRADE,
            content=msg,
            timestamp=timestamp
        )

    @classmethod
    def trade_message_from_db(cls, record: RowProxy, metadata: Optional[Dict] = None):
        """
        *used for backtesting
        Convert a row of trade data into standard OrderBookMessage format
        :param record: a row of trade data from the database
        :return: BinarzOrderBookMessage
        """
        return BinarzOrderBookMessage(
            message_type=OrderBookMessageType.TRADE,
            content=record.json,
            timestamp=record.timestamp
        )

    @classmethod
    def from_snapshot(cls, snapshot: OrderBookMessage):
        raise NotImplementedError(constants.EXCHANGE_NAME + " order book needs to retain individual order data.")

    @classmethod
    def restore_from_snapshot_and_diffs(self, snapshot: OrderBookMessage, diffs: List[OrderBookMessage]):
        raise NotImplementedError(constants.EXCHANGE_NAME + " order book needs to retain individual order data.")
