#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import (print_function, unicode_literals,
                        absolute_import, with_statement)

import os
import sys


if __name__ == '__main__':
    if __package__ is None:
        dir_name = os.path.dirname(__file__)
        sys.path.append(
            os.path.abspath(
                os.path.join(dir_name, '..')))

    from excel2mysql.migrate import migrate

    migrate()
