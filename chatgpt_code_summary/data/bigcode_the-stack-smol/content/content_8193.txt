"""Several tools used accross by other modules"""

import logging
from logging.handlers import BufferingHandler
from asyncio import sleep, get_event_loop
from datetime import datetime, timedelta
from distutils.util import strtobool
from os.path import abspath, dirname
from typing import Union, Optional, List
from uuid import uuid4
import jwt as jwtlib

logger = logging.getLogger(__name__)


def find(func, iteratee):
    """Returns the first element that match the query"""
    for value in iteratee:
        if func(value):
            return value
    return None


def cast(val, typ, *types):
    """Cast a value to the given type. /!\\ Hack /!\\ """

    # get Optional
    if typ.__class__ in [Union.__class__, Optional.__class__] \
       and len(typ.__args__) == 2 \
       and typ.__args__[1] is None:
        typ = typ.__args__[0]

    # split Unions
    elif typ.__class__ == Union.__class__:
        return cast(val, *typ.__args__)

    # consume List
    if typ.__class__ == List.__class__:
        values = []
        for element in val:
            values.append(cast(element, typ.__args__[0]))
        return values

    # cast
    types = [typ] + list(types)
    for typ in types:
        try:
            return typ(val)
        except:
            continue

    raise TypeError("{} not castable in any of {{{}}}.".format(val, types))


def real_type(typ):
    """Escape the type from Union and Optional. /!\\ Hack /!\\ """
    if typ.__class__ in [Union.__class__, Optional.__class__]:
        return typ.__args__[0]
    return typ


def root():
    """Return the path of the package root"""
    return dirname(abspath(__file__))


class DelayLogFor(BufferingHandler):
    """Delai logging for a specific logger."""
    def __init__(self, delayed_logger: logging.Logger):
        self.delayed_logger = delayed_logger
        self.delayed_handlers = []
        super().__init__(float('infinity'))

    def flush(self):
        """Flush this BufferingHandler to all the delayed handlers."""
        self.acquire()
        try:
            for handler in self.delayed_handlers:
                for record in self.buffer:
                    if record.levelno >= handler.level:
                        handler.handle(record)
            self.buffer = []
        finally:
            self.release()

    def __enter__(self):
        """Replace the handlers by this BufferingHandler"""
        self.delayed_handlers.extend(self.delayed_logger.handlers)
        self.delayed_logger.handlers.clear()
        self.delayed_logger.addHandler(self)
        return self

    def __exit__(self, typ, val, traceback):
        """Restore the handlers and flush this BufferingHandler"""
        self.delayed_logger.removeHandler(self)
        self.delayed_logger.handlers.extend(self.delayed_handlers)
        self.close()


def generate_token(key, iat=None, exp_delta=timedelta(minutes=5), typ="player",
                   tid=None, uid="00000000-0000-0000-0000-000000000000"):
    """Generate a JSON Web Token"""
    if iat is None:
        iat = datetime.utcnow()

    if tid is None:
        tid = str(uuid4())

    return jwtlib.encode({
        "iss": "webapi",
        "sub": "webgames",
        "iat": iat,
        "exp": iat + exp_delta,
        "jti": tid,
        "typ": typ,
        "uid": uid
    }, key, algorithm='HS256').decode()


def ask_bool(prompt):
    """Ask a question to the user, retry until the reply is valid"""
    while True:
        try:
            return strtobool(input("%s (yes/no) " % prompt).strip().casefold())
        except ValueError:
            continue


def fake_async(func):
    """Fake coroutine by awaiting asyncio.sleep(0)"""
    async def wrapped(*args, **kwargs):
        """The faked coroutine"""
        await sleep(0)
        return func(*args, **kwargs)
    return wrapped


def lruc(coro, loop=get_event_loop()):
    """Short version of loop.run_until_complete(coro)"""
    return loop.run_until_complete(coro)


def async_partial(func, *args, **keywords):
    """async functools.partial"""
    async def newfunc(*fargs, **fkeywords):
        """the mocked function"""
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return await func(*args, *fargs, **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc

class Ref:
    _obj = None
    def __call__(self, obj):
        self._obj = obj

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __setattr__(self, attr, value):
        if attr == "_obj":
            super().__setattr__(attr, value)
        return setattr(self._obj, attr, value)
