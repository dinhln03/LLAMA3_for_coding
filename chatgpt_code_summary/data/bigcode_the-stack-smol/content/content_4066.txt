# coding=utf-8
from __future__ import print_function
import json

from data_packer import err, DataPacker, container


g_src = {
    'a': 1,
    'b': 'hello',
    'c': ['a', 'b', 'c'],
    'd': {
        '1': 1,
        '2': 2,
    },
    'e': {
        '1': ['a', 'b'],
        '2': {
            'a': 'a',
            'b': 'b'
        }
    },
    'f': '0x123',
    'g': 'longlonglonglonglong',
    'h': 2,
}


def valid_container(c):
    if isinstance(c, dict):
        c = container.DictContainer(c)
    else:
        raise TypeError('dst Must be dict or DictContainer')

    return c

def demo_run(fields, msg, dst=None, src=None):
    print('')
    print(msg)

    if src is None:
        src = g_src
    if dst is None:
        dst = {}

    src = valid_container(src)
    dst = valid_container(dst)

    dp = DataPacker(fields)
    try:
        dp.run(src, dst)
    except err.DataPackerError as e:
        print('抛出了异常: ', type(e), e)

    print(json.dumps(dst.raw_data(), indent=4))

    return dst

