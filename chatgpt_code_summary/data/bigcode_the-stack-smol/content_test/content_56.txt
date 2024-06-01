import pytest

from temp import (download_to_file, ensure_datafile, records_from_lines, make_record, make_value, min_spread_record,
                 min_spread_day_num, parse_header)


def test_download_to_file(tmpdir):
    file = tmpdir.join('test.txt')
    download_to_file(file.strpath, 'https://httpbin.org/get?testParam=1')
    assert 'testParam' in file.read()


def test_ensure_datafile_downloads(tmpdir):
    file = tmpdir.join('test.txt')
    ensure_datafile(file.strpath, 'https://httpbin.org/get?testParam=1')
    assert 'testParam' in file.read()


def test_ensure_datafile_uses_existing(tmpdir):
    file = tmpdir.join('test.txt')
    file.write('content')
    ensure_datafile(file.strpath, 'https://httpbin.org/get?testParam=1')
    assert file.read() == 'content'


def test_make_record():
    header = {'One': (0, 1), 'Two': (3, 3), 'Three': (7, 2), 'Four': (10, 4)}
    line = "1 2.2 3* FOUR"
    rv = make_record(header, line)
    assert set(rv.keys()) == set(header)
    assert all(list(rv.values()))


def test_parse_header_rv():
    rv = parse_header(" One Two Three")
    assert len(rv) == 3
    assert rv['One'] == (1, 4)
    assert rv['Two'] == (5, 4)
    assert rv['Three'] == (9, 6)


def test_make_value():
    assert make_value('123') == 123
    assert make_value('ASDF') == 'ASDF'
    assert make_value('123.45') == 123.45
    assert make_value('123*') == 123
    assert make_value(' ') == None


def test_records_from_lines_skips_empty():
    lines = iter(['One', ' ', 'Two'])
    assert len(list(records_from_lines(lines))) == 1


def test_min_spread_record_rv():
    records = [
        {'max': 10, 'min': 0},
        {'max': 1, 'min': 0},   # this one should be returned
        {'max': None, 'min': None}
    ]
    assert min_spread_record(records, 'max', 'min') == {'max': 1, 'min': 0}


def test_min_spread_day_num_rv():
    records = [
        {'Dy': 1, 'MxT': 10, 'MnT': 0},
        {'Dy': 2, 'MxT': 5, 'MnT': 0},
    ]
    rv = min_spread_day_num(records)
    assert rv == 2
