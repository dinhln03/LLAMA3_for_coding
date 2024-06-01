
import logging
import pyrebase

from requests.exceptions import HTTPError

class Node:
    def __init__(self, nodeName):
        self._nodeName = nodeName
        self._next = None

    def child(self, nodeName):
        if self._next == None:
            self._next = Node(nodeName)
        else:
            self._next.next(nodeName)
        return self

    def set(self, data):
        if self._next == None:
            self._next = Set(data)
        else:
            self._next.set(data)
        return self

    def get(self):
        if self._next == None:
            self._next = Get()
        else:
            self._next.get()
        return self

    def eval(self, prev):
        return self._next.eval(prev.child(self._nodeName))

    def __str__(self):
        if self._next == None:
            return 'child(' + str(self._nodeName) + ')'
        else:
            return 'child(' + str(self._nodeName) + ').' + str(self._next)

class Set:
    def __init__(self, data):
        self._data = data

    def eval(self, prev):
        return prev.set(self._data)

    def __str__(self):
        return 'set(' + str(self._data) + ')'

class Get:
    def eval(self, prev):
        return prev.get()

    def __str__(self):
        return 'get()'

class Remove:
    def eval(self, prev):
        return prev.remove()

    def __str__(self):
        return 'remove()'

class Push:
    def __init__(self, data):
        self._data = data

    def eval(self, prev):
        return prev.push(self._data)

    def __str__(self):
        return 'push(' + str(self._data) + ')'

class Update:
    def __init__(self, data):
        self._data = data

    def eval(self, prev):
        return prev.update(self._data)

    def __str__(self):
        return 'update(' + str(self._data) + ')'

class FirebaseLiveEvaluator:
    def __init__(self, config):
        logging.info('Initializing Firebase connection...')
        self._firebase = pyrebase.initialize_app(config)
        self._db = self._firebase.database()
        self._pathPrefix = config['firebasePathPrefix']

    def eval(self, node):
        # logging.debug(node)
        if self._pathPrefix:
            return node.eval(self._db.child(self._pathPrefix))
        else:
            return node.eval(self._db)

class FirebaseLoggingEvaluator:
    def eval(self, node):
        logging.info(node)

class FirebaseExceptionEvaluator:
    def __init__(self, config):
        logging.info('Initializing Firebase connection...')
        self._firebase = pyrebase.initialize_app(config)
        self._db = self._firebase.database()
        self._pathPrefix = config['firebasePathPrefix']
        self._throw = True

    def eval(self, node):
        if self._throw:
            self._throw = False
            raise HTTPError("I Broke")

        logging.debug(node)
        if self._pathPrefix:
            return node.eval(self._db.child(self._pathPrefix))
        else:
            return node.eval(self._db)
