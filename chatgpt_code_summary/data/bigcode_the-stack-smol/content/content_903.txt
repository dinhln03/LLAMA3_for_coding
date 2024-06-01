# -*- coding: utf-8 -*-

import sys
import argparse

from cgate.reader import readfile, readschema, get_dtype
from cgate.validation import validate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', help='Table name or File path')
    parser.add_argument('--schema', '-s', help='Cerberus schema file')
    parser.add_argument('--null', '-n', help='Null character', default='NULL,\\N')
    args = parser.parse_args()

    schema = readschema(args.schema)
    try:
        header = schema['header']
    except:
        header = None
    na_values = args.null.split(',')
    dtype, date_cols = get_dtype(schema['schema'])
    dfs = readfile(args.target, header=header, dtype=dtype, parse_dates=date_cols, na_values=na_values)
    fail_count = validate(dfs, schema['schema'])
    if fail_count != 0:
        print('Failed {0} error...'.format(fail_count), file=sys.stderr)
        return 1
    print('Success!', file=sys.stderr)
    return 0
