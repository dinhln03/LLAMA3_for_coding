# -*- coding: utf-8 -*-
'''
Runs MultiprocessTest with all warnings including traceback...
'''
#
# https://stackoverflow.com/questions/22373927/get-traceback-of-warnings
import traceback
import warnings
import sys

from . import multiprocess


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def main(test_group=None):
    warnings.showwarning = warn_with_traceback
    warnings.simplefilter("always")
    multiprocess.main(test_group)


if __name__ == '__main__':
    main()
