import json
import re
from types import FunctionType, LambdaType

_pattern_type = re.compile('t').__class__


class ConversionIndex(object):
    def __init__(self, index_name=''):
        self.o2t = {}
        self.t2o = {}
        self.name = index_name

    def set_index(self, origin_index, target_index):
        if isinstance(origin_index, str):
            self.o2t[origin_index] = target_index
            self.t2o[target_index] = origin_index
        elif isinstance(origin_index, list):
            self.o2t['-'.join(origin_index)] = target_index
            self.t2o[target_index] = origin_index

    def get_origin_by_target(self, target_index):
        return self.t2o[target_index]

    def is_origin_indexed(self, origin_index):
        if isinstance(origin_index, str):
            return origin_index in self.o2t
        elif isinstance(origin_index, list):
            return '-'.join(origin_index) in self.o2t

    def get_target_by_origin(self, origin_index):
        if isinstance(origin_index, str):
            return self.o2t[origin_index]
        elif isinstance(origin_index, list):
            return self.o2t['-'.join(origin_index)]
        else:
            return None

    def is_target_indexed(self, target_index):
        return target_index in self.t2o

    def reset(self):
        self.o2t = None
        self.t2o = None
        self.name = None

    def dump(self, file_name):
        obj = {
            'name': self.name,
            'o2t':  self.o2t,
            't2o':  self.t2o,
        }
        with open(file_name, 'w') as fp:
            json.dump(obj, fp)

    @staticmethod
    def load(file_name):
        with open(file_name, 'r') as fp:
            idx = ConversionIndex()
            obj = json.load(fp)
            idx.name = obj['name']
            idx.o2t = obj['o2t']
            idx.t2o = obj['t2o']

        return idx


class FieldIndex(object):
    def __init__(self, table):
        self.table = table
        self.indices = {}

    def create_index(self, name, **kwargs):
        index = {}
        keys = [key for key in kwargs.keys() if key in self.table.headers()]
        is_callable = {}
        is_regex = {}

        for key in keys:
            index[key] = []
            value = kwargs[key]
            is_callable[key] = isinstance(value, FunctionType) or isinstance(value, LambdaType)
            is_regex[key] = isinstance(value, _pattern_type)

        for idx, row in enumerate(self.table):
            for key in keys:
                value = kwargs[key]
                if value:
                    if is_callable[key]:
                        index_available = value(idx, row)
                    elif is_regex[key]:
                        index_available = bool(value.match(row[key]))
                    else:
                        index_available = row[key] == value
                    if not index_available:
                        continue
                index[key].append(idx)

        for key in keys:
            index[key] = set(index[key])

        self.indices[name] = index

        return self

    def get_index(self, name, *args):
        if len(args) == 0 or name not in self.indices:
            return None

        index = self.indices[name]
        result = index[args[0]]

        for key in args[1:]:
            result = result.intersection(index[key])

        return result

    def filter_index(self, name, filters, dict_key=None):
        intersect = self.get_index(name, *filters.keys())

        is_callable = {}
        is_regex = {}

        for key in filters:
            value = filters[key]
            is_callable[key] = isinstance(value, FunctionType) or isinstance(value, LambdaType)
            is_regex[key] = isinstance(value, _pattern_type)

        if dict_key:
            output = {}
        else:
            output = []

        for idx in intersect:
            row = self.table.row(idx)
            available = True
            for key, value in filters.items():
                if value:
                    if is_callable[key]:
                        available = value(idx, row)
                    elif is_regex[key]:
                        available = bool(value.match(row[key]))
                    else:
                        available = row[key] == value
                    if not available:
                        break
                else:
                    available = False
                    break

            if available:
                if dict_key:
                    if row[dict_key] not in output:
                        output[row[dict_key]] = []
                    output[row[dict_key]].append(idx)
                else:
                    output.append(idx)

        return output
