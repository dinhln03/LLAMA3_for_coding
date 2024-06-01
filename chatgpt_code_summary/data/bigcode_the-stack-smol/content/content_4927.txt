import pandas as pd
import numpy as np
import itertools as it
from collections import defaultdict
from collections import Counter
from six.moves import map as imap


def dict_subset(d, fields):
    # return a subset of the provided dict containing only the
    # fields specified in fields
    return {k: d[k] for k in d if k in fields and d[k] is not None}


class MessageFieldCounter:
    """
    Count occurrences of values in a stream of messages for a specified set of fields
    Usage:

       messages = [
             {'a': 'apple', 'b': 'boat'},
             {'a': 'pear', 'b': 'boat'},
             {'a': 'apple', 'b': 'boat'},
             ]
       fields = ['a', 'b']
       mfc = MessageFieldCounter(messages, fields)
       # this class is designed to pass through a long stream of messages
       # so we have to pull them through in order to count them
       for msg in mcf:
              pass
        print mfc.most_common('a')
        >>> [('apple', 2)]
    """

    def __init__(self, messages, fields):
        self.fields = set(fields)
        self.messages = messages
        self.counters = defaultdict(Counter)

    def __iter__(self):
        return self.process()

    def process(self):
        for msg in self.messages:
            for key in self.fields:
                value = msg.get(key, None)
                if value is not None:
                    self.counters[key][value] += 1

            yield msg

    def most_common(self, field, n=1):
        return self.counters[field].most_common(n)


class MessageStats():
    """
    Extract a set of stats from as stream of messages.

    numeric_fields: list of field names to compute numeric stats (eg min, max, avg)
    frequency_fields: list of field names to compute frequency of values
    """
    NUMERIC_STATS = ['min', 'max', 'first', 'last', 'count']
    FREQUENCY_STATS = ['most_common', 'most_common_count']

    def __init__(self, messages, numeric_fields, frequency_fields):
        self._numeric_fields = numeric_fields
        self._frequency_fields = frequency_fields

        self.counter = MessageFieldCounter(messages, frequency_fields)
        messages = self.counter.process()
        messages = imap(dict_subset, messages, it.repeat(numeric_fields))
        # DataFrame won't take an iterator, but it will take a generator
        messages = (m for m in messages)
        self.df = pd.DataFrame(messages)

    @property
    def numeric_fields(self):
        return self._numeric_fields

    @property
    def frequency_fields(self):
        return self._frequency_fields

    @property
    def frequency_counter(self):
        return self.counter

    @property
    def data_frame(self):
        return self.df

    def numeric_stats(self, field):
        def first(col):
            idx = col.first_valid_index()
            return col[idx] if idx is not None else None

        def last(col):
            idx = col.last_valid_index()
            return col[idx] if idx is not None else None

        assert field in self.numeric_fields
        if field in self.df:
            col = self.df[field]
            return dict(
                min=np.nanmin(col),
                max=np.nanmax(col),
                first=first(col),
                last=last(col),
                count=np.count_nonzero(~np.isnan(col)),
            )
        else:
            return {}

    def frequency_stats(self, field):
        assert field in self.frequency_fields
        stat = self.frequency_counter.most_common(field)
        if stat:
            value, count = stat[0]
            return dict(
                most_common=value,
                most_common_count=count
            )
        else:
            return {}

    def field_stats(self, field):
        stats = {}
        if field in self.numeric_fields:
            stats.update(self.numeric_stats(field))
        if field in self.frequency_fields:
            stats.update(self.frequency_stats(field))
        return stats