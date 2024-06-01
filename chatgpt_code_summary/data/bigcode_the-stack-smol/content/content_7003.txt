import pickle
import zmq
import threading
import pymongo
from constants import (
    MONGO_DEFAULT_HOST,
    MONGO_DEFAULT_PORT,
    ZMQ_DEFAULT_HOST,
    ZMQ_DEFAULT_PORT
)

class QueryExecutor(object):
    """A query executor"""

    def __init__(self, dbconfig={}, zmqconfig={}):
        """Initialize executor"""
        self.dbconfig = dbconfig
        self.zmqconfig = zmqconfig
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._conn = None

    def _connectdb(self):
        """Connects Mongo"""
        self._conn = pymongo.MongoClient(self.zmqconfig.get('host', MONGO_DEFAULT_HOST), self.zmqconfig.get('port', MONGO_DEFAULT_PORT))

    def _listen(self):
        """Start listening for queries"""
        # start server
        self._socket.bind('tcp://{}:{}'.format(self.zmqconfig.get('host', ZMQ_DEFAULT_HOST), self.zmqconfig.get('port', ZMQ_DEFAULT_PORT)))
        while True:
            query = self._socket.recv_pyobj()
            self._socket.send_pyobj('QUERY_RECEIVED')
            callback_string = self._socket.recv_pyobj()
            callback = pickle.loads(callback_string)
            self._socket.send_pyobj('CALLBACK_RECEIVED')
            self._execute(query, callback)

    def _execute(self, query, callback):
        """Return the query result"""
        if not self._conn:
            self._connectdb()
        query_result = self._conn.[query['dbname']][query['colname']]
        return callback(query_result)

    def start(self):
        "Start executor thread"
        thread = threading.Thread(target=self._listen, args=())
        thread.daemon = True
        thread.start()
