# coding=utf-8

import websocket
import datetime
import csv
import time
import logging
import redis
import json
import copy
import pytz
from hftcoin.mdagent.ccws.configs import REDIS_HOST
from hftcoin.mdagent.ccws.configs import TIMEZONE
from hftcoin.mdagent.ccws.configs import ExConfigs
from hftcoin.mdagent.ccws.configs import HOME_PATH


class Exchange(object):
    ExchangeId = ''

    WebSocketConnection = None
    RedisConnection = None

    def __init__(self):
        self.Logger = logging.getLogger(self.ExchangeId)
        [self.ExConfig, self._WebSocketAddress] = ExConfigs[self.ExchangeId]
        self.Config = {}

    def set_market(self, currency, mode):
        self.Config = self.ExConfig[currency][mode]
        self.Logger = logging.getLogger('%s.%s.%s' % (self.ExchangeId, currency, mode))

    def run_websocketapp(self, **kwargs):
        self.Logger.info('Begin Connection')
        url = self._WebSocketAddress + kwargs.pop('url_append', '')
        on_error = kwargs.pop('on_error', self.on_error)
        on_close = kwargs.pop('on_close', self.on_close)
        on_message = kwargs.pop('on_message', self.on_message)
        self.WebSocketConnection = websocket.WebSocketApp(
            url,
            on_error=on_error,
            on_close=on_close,
            on_message=on_message,
            **kwargs,
        )
        while True:
            try:
                self.WebSocketConnection.run_forever()
            except Exception as e:
                self.Logger.exception(e)

    def on_message(self, _ws, msg):
        ts = int(time.time()*1000)
        rdk = self.Config['RedisCollectKey']
        # self.Logger.debug(msg)
        self.RedisConnection.lpush(rdk, json.dumps([ts, msg]))

    def on_error(self, _ws, error):
        self.Logger.exception(error)

    def on_close(self, _ws):
        self.Logger.info('Connection closed.')

    def connect_redis(self):
        try:
            self.RedisConnection = redis.StrictRedis(host=REDIS_HOST)
            self.RedisConnection.ping()
        except Exception as e:
            self.Logger.exception(e)

    def write_data_csv(self):
        self.connect_redis()
        [fn, rdk] = [self.Config.get(item) for item in ['FileName', 'RedisOutputKey']]
        error_count = 100
        while True:
            try:
                if self.RedisConnection.llen(rdk) > 0:
                    data = json.loads(self.RedisConnection.rpop(rdk).decode('utf8'))
                    # data[1] is timestamp
                    dt = datetime.datetime.fromtimestamp(data[1] / 1000, TIMEZONE)
                    calendar_path = '%4d/%02d/%02d' % (dt.year, dt.month, dt.day)
                    with open('%s/%s/%s' % (HOME_PATH, calendar_path, fn), 'a+') as csvFile:
                        csvwriter = csv.writer(csvFile)
                        csvwriter.writerow(data)
                else:
                    time.sleep(60)
            except RuntimeWarning:
                break
            except Exception as e:
                self.Logger.exception(e)
                error_count -= 1
                if error_count < 0:
                    break

    def collect_data(self):
        pass

    def process_data(self):
        self.connect_redis()
        getattr(self, self.Config.get('DataHandler', object))()

    def _check_price_eq(self, p1, p2):
        # divide by 2 to avoid precision
        return abs(p1-p2) < self.Config['TickSize']/2

    def _binary_search(self, find, list1, low, high):
        while low <= high:
            mid = int((low + high) / 2)
            if self._check_price_eq(list1[mid][0], find):
                return [mid, 'True']
            elif list1[mid][0] > find:
                high = mid - 1
            else:
                low = mid + 1
        return [low, 'False']

    def _update_order_book(self, bids, asks, side, price, remaining):
        if side in ['bid', 'buy']:
            book = bids
            cut = int(99*(len(book)-1)/100)
        else:
            book = asks
            cut = int((len(book)-1)/100)

        if price < book[cut][0]:
            res = self._binary_search(price, book, 0, cut-1)
        else:
            res = self._binary_search(price, book, cut, len(book)-1)

        if res[1] == 'True':
            if remaining < self.Config['AmountMin']:
                del book[res[0]]
            else:
                book[res[0]][1] = remaining
        else:
            if remaining >= self.Config['AmountMin']:
                book.insert(res[0], [price, remaining])

    def check_data_validation(self, book):
        length = int(len(book)/2)
        for i in range(0, length - 2, 2):
            if book[i] <= book[i + 2]:
                return False
        for i in range(length, 2 * length - 2, 2):
            if book[i] >= book[i + 2]:
                return False
        for i in range(1, 2 * length, 2):
            if book[i] < self.Config['AmountMin']:
                return False
        if book[0] > book[length]:
            return False
        return True

    @staticmethod
    def _cut_order_book(bids, asks, depth):
        if len(bids) >= depth:
            book = bids[-depth:]
            book.reverse()
        else:
            book = copy.deepcopy(bids)
            book.reverse()
            book += [['None', 'None']] * (depth - len(bids))

        if len(asks) >= depth:
            book += asks[:depth]
        else:
            book += asks + [['None', 'None']] * (depth - len(asks))
        book = [x[0:2] for x in book]

        return sum(book, [])

    @staticmethod
    def fmt_date(ts):
        return datetime.datetime.fromtimestamp(ts / 1000, TIMEZONE).strftime('%Y-%m-%d %H:%M:%S.%f %z')

    @staticmethod
    def date_from_str(ts):
        return pytz.utc.localize(datetime.datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%fZ'))
