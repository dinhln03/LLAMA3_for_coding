"""Test cases for the pypfilt.io module."""

import datetime
import numpy as np
import os

from pypfilt.io import read_table, date_column


def test_read_datetime():
    # Test data: sequential dates with Fibonacci sequence.
    content = """
    date count
    2020-01-01 1
    2020-01-02 1
    2020-01-03 2
    2020-01-04 3
    2020-01-05 5
    2020-01-06 8
    2020-01-07 13
    2020-01-08 21
    2020-01-09 34
    """
    expect_rows = 9
    expect_count = [1, 1]
    for i in range(expect_rows - 2):
        expect_count.append(expect_count[i] + expect_count[i + 1])

    # Save this data to a temporary data file.
    path = "test_read_datetime.ssv"
    with open(path, encoding='utf-8', mode='w') as f:
        f.write(content)

    # Read the data and then remove the data file.
    columns = [
        date_column('date'),
        ('count', np.int_),
    ]
    df = read_table(path, columns)
    os.remove(path)

    # Check that we received the expected number of rows.
    assert len(df) == expect_rows

    # Check that each row has the expected content.
    for ix, row in enumerate(df):
        assert isinstance(row['date'], datetime.datetime)
        assert row['date'].year == 2020
        assert row['date'].month == 1
        assert row['date'].day == ix + 1
        assert row['count'] == expect_count[ix]
