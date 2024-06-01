# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Constant values used by this library.
"""

from enum import Enum


class DataCategory(Enum):
    """
    Enumeration of data categories in compliant machine learning.

    Values:
    - PRIVATE: data which is private. Researchers may not view this.
    - PUBLIC: data which may safely be viewed by researchers.
    """

    PRIVATE = 1
    PUBLIC = 2
