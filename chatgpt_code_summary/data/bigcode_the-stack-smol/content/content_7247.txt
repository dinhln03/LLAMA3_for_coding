import collections
from typing import Iterator

import itertools

from stream_lib.stream_api import Stream, T


class ItertoolsStream(Stream[T]):
    @staticmethod
    def stream(*iterables: Iterator[T]):
        if len(iterables) == 1:
            return ItertoolsStream(*iterables)
        else:
            return ItertoolsStream(itertools.zip_longest(*iterables))

    def __init__(self, delegate: Iterator[T]):
        assert isinstance(delegate, collections.Iterable)
        if not isinstance(delegate, collections.Iterator):
            delegate = iter(delegate)
        self._delegate = delegate

    def __iter__(self):
        self._delegate = iter(self._delegate)
        return self._delegate

    def __next__(self):
        return next(self._delegate)

    def map(self, func):
        return self._stream(map(func, self))

    def flatmap(self, func):
        return self.map(func).flatten()

    def flatten(self):
        return self._stream(itertools.chain.from_iterable(self))

    def filter(self, predicate):
        return self._stream(filter(predicate, self))

    def slice(self, start, stop, step=1):
        return self._stream(itertools.islice(self, start, stop, step))

    def limit(self, size):
        return self.slice(0, size)
