import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
        '-p',
        dest='path',
        help='Spectify the path')
parser.add_argument(
        '-l', action='store_true',
        dest='long_format',
        help='use a long listing format')
parser.add_argument(
        '-a', action='store_true',
        dest='show_hidden',
        help='do not ignore entries starting with .')
parser.add_argument(
        '-S', action='store_true',
        dest='sort_by_size',
        help='sort by file size')
parser.add_argument(
        '-R', action='store_true',
        dest='list_subdir',
        help='list subdirectories recursively')

args = parser.parse_args()
if not args.path:
    args.path = os.getcwd()
