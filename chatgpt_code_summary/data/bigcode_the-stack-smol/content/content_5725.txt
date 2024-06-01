# -*- coding: utf-8 -*-
"""Test human2bytes function."""

import pytest
from pcof import bytesconv


@pytest.mark.parametrize(
    "size, unit, result",
    [
        (1, "KB", "1024.00"),
        (1, "MB", "1048576.00"),
        (1, "GB", "1073741824.00"),
        (1, "TB", "1099511627776.00"),
        (1, "PB", "1125899906842624.00"),
        (1, "EB", "1152921504606846976.00"),
    ],
)
def test_human2bytes(size, unit, result):
    assert bytesconv.human2bytes(size, unit) == result


@pytest.mark.parametrize(
    "size, unit, precision, result",
    [
        (1, "KB", 0, "1024"),
        (2, "GB", 0, "2147483648"),
        (2, "GB", 1, "2147483648.0"),
        (2, "GB", 3, "2147483648.000"),
    ],
)
def test_human2bytes_precision(size, unit, precision, result):
    assert bytesconv.human2bytes(size, unit, precision=precision) == result


@pytest.mark.parametrize(
    "size, unit, base, result",
    [
        (1, "KB", 1000, "1000.00"),
        (1, "MB", 1000, "1000000.00"),
        (1, "GB", 1000, "1000000000.00"),
        (1, "TB", 1000, "1000000000000.00"),
        (4, "TB", 1000, "4000000000000.00"),
        (1, "PB", 1000, "1000000000000000.00"),
        (1, "EB", 1000, "1000000000000000000.00"),
    ],
)
def test_human2bytes_base(size, unit, base, result):
    assert bytesconv.human2bytes(size, unit, base=base) == result


def test_human2bytes_raise():
    with pytest.raises(ValueError, match="value is not a number"):
        bytesconv.human2bytes("notnumber", "KB")
    with pytest.raises(
        ValueError, match="invalid unit. It must be KB, MB, GB, TB, PB, EB, ZB"
    ):
        bytesconv.human2bytes(1, "XX")


# vim: ts=4
