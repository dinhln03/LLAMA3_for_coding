#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of Archdiffer and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php

"""
Created on Sun Mar  4 10:23:41 2018

@author: Pavla Kratochvilova <pavla.kratochvilova@gmail.com>
"""

import operator
import datetime
from flask import request
from .exceptions import BadRequest

def make_datetime(time_string, formats=None):
    """Makes datetime from string based on one of the formats.

    :param string time_string: time in string
    :param list formats: list of accepted formats
    :return datetime.datetime: datetime or None if no format is matched
    """
    if formats is None:
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
        ]
    for fmt in formats:
        try:
            return datetime.datetime.strptime(time_string, fmt)
        except ValueError:
            pass
    return None

# Transformation functions
def _dict_transform(string):
    return dict([item.split(':', 1) for item in string.split(';')])

def _list_transform(string):
    return string.split(',')

# Transformations of common arguments
_TRANSFORMATIONS = {
    'filter_by' : _dict_transform,
    'filter' : _list_transform,
    'order_by' : _list_transform,
    'limit' : lambda x: int(x),
    'offset' : lambda x: int(x),
}

# Filters creators
def before(column, name='before'):
    """Make filter template for filtering column values less or equal to
    datetime.

    :param column: database model
    :param string name: name used in the filter template
    :return dict: resulting template
    """
    return {name: (column, operator.le, make_datetime)}

def after(column, name='after'):
    """Make filter template for filtering column values greater or equal to
    datetime.

    :param column: database model
    :param string name: name used in the filter template
    :return dict: resulting template
    """
    return {name: (column, operator.ge, make_datetime)}

def time(column, name='time'):
    """Make filter template for filtering column values equal to datetime.

    :param column: database model
    :param string name: name used in the filter template
    :return dict: resulting template
    """
    return {name: (column, operator.eq, make_datetime)}

def equals(column, name='id', function=(lambda x: x)):
    """Make filter template for filtering column values equal to value
    transformed by given function.

    :param column: database model
    :param string name: name used in the filter template
    :param callable function: function for transforming the value
    :return dict: resulting template
    """
    return {name: (column, operator.eq, function)}

# Request parser
def parse_request(filters=None, defaults=None):
    """Parse arguments in request according to the _TRANSFORMATIONS or given
    filters.
    Requests containing other keys are considered invalid.

    :param dict filters: dict of filter templates containing for each key
        (column, operator, function transforming value from request argument)
    :param dict defaults: default values of modifiers
    :return dict: dict of parsed arguments
    :raises werkzeug.exceptions.BadRequest: if one of the request arguments is
        not recognized
    """
    if filters is None:
        filters = {}
    if defaults is not None:
        args_dict = defaults.copy()
    else:
        args_dict = {}
    filters_list = []

    for key, value in request.args.items():
        if key in _TRANSFORMATIONS:
            try:
                args_dict[key] = _TRANSFORMATIONS[key](value)
            except ValueError:
                raise BadRequest('Argument has invalid value "%s".' % value)
        elif key in filters.keys():
            filters_list.append(
                filters[key][1](filters[key][0], filters[key][2](value))
            )
        else:
            raise BadRequest('Argument "%s" not recognized.' % key)

    if 'filter' not in args_dict.keys():
        args_dict['filter'] = []
    args_dict['filter'] += filters_list

    return args_dict

def get_request_arguments(*names, args_dict=None, invert=False):
    """Get arguments from args_dict or request if they match given names.

    :param *names: names of arguments
    :param dict args_dict: dict of arguments
    :param bool invert: True if names should be exclueded instead
    :return dict: dict of arguments
    """
    if args_dict is None:
        args_dict = parse_request()
    if invert:
        return {k:v for k, v in args_dict.items() if k not in names}
    return {k:v for k, v in args_dict.items() if k in names}

def update_modifiers(old_modifiers, new_modifiers):
    """Update modifiers.

    :param dict old_modifiers: old modifiers
    :param dict old_modifiers: new modifiers
    :return dict: resulting modifiers
    """
    modifiers = old_modifiers.copy()
    for key, value in new_modifiers.items():
        if key in old_modifiers:
            if _TRANSFORMATIONS.get(key) == _list_transform:
                modifiers[key] += value
            elif _TRANSFORMATIONS.get(key) == _dict_transform:
                modifiers[key].update(value)
            else:
                modifiers[key] = value
        else:
            modifiers[key] = value
    return modifiers
