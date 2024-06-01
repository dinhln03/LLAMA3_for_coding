import pytest
import math
import os
import sys

module_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(module_dir, '..', 'intervalpy'))
from intervalpy import Interval

def test_intersection():
    # closed, closed
    d1 = Interval(0, 2, start_open=False, end_open=False)
    d2 = Interval(1, 3, start_open=False, end_open=False)

    assert d1.contains(0)
    assert d1.contains(1)
    assert d1.contains(2)

    d = Interval.intersection([d1, d2])
    assert d.start == 1
    assert d.end == 2
    assert not d.start_open
    assert not d.end_open

    d = Interval.union([d1, d2])
    assert d.start == 0
    assert d.end == 3
    assert not d.start_open
    assert not d.end_open

    # closed, open
    d1 = Interval(0, 2, start_open=False, end_open=False)
    d2 = Interval(1, 3, start_open=True, end_open=True)

    d = Interval.intersection([d1, d2])
    assert d.start == 1
    assert d.end == 2
    assert d.start_open
    assert not d.end_open

    d = Interval.union([d1, d2])
    assert d.start == 0
    assert d.end == 3
    assert not d.start_open
    assert d.end_open

    # open, open
    d1 = Interval(0, 2, start_open=True, end_open=True)
    d2 = Interval(1, 3, start_open=True, end_open=True)

    assert not d1.contains(0)
    assert d1.contains(1)
    assert not d1.contains(2)

    d = Interval.intersection([d1, d2])
    assert d.start == 1
    assert d.end == 2
    assert d.start_open
    assert d.end_open

    d = Interval.union([d1, d2])
    assert d.start == 0
    assert d.end == 3
    assert d.start_open
    assert d.end_open

    d = Interval.intersection([Interval(0, 1), Interval(2, 3)])
    assert d.is_empty

    d = Interval.intersection([Interval(0, 1, end_open=True), Interval(1, 3, start_open=True)])
    assert d.is_empty

    d = Interval.intersection([Interval(0, 1), Interval.empty()])
    assert d.is_empty

    d = Interval.union([Interval.empty(), 1])
    assert d.start == 1
    assert d.end == 1

def test_interval_contains_inf():
    inf = Interval.infinite()
    assert inf.contains(math.inf) is True
    assert inf.contains(-math.inf) is True
    assert Interval.gte(0).contains(math.inf) is True
    assert Interval.gte(0).contains(-math.inf) is False
    assert Interval.lte(0).contains(math.inf) is False
    assert Interval.lte(0).contains(-math.inf) is True

def test_intersection_inf():
    assert Interval.intersection([Interval.gte(100), (98, 101)]) == (100, 101)
    assert Interval.intersection([Interval.point(100), Interval.open_closed(100, 101)]) == Interval.empty()

def test_cast():
    assert bool(Interval.empty()) is False
    assert bool(Interval(0, 0)) is True
    assert list(Interval.empty()) == []
    assert list(Interval(0, 0)) == [0, 0]
    assert list(Interval.open(1, 20)) == [1, 20]

def test_intersects():
    assert Interval.closed(1, 3).intersects(Interval.closed(2, 3))
    assert Interval.closed(1, 3).intersects((2, 3))
    assert Interval.closed(1, 3).intersects((1, 3))
    assert Interval.closed(1, 3).intersects(Interval.open(1, 3))

    assert Interval.closed(1, 3).intersects(Interval.closed(3, 4))
    assert not Interval.closed(1, 3).intersects(Interval.open(3, 4))
    assert not Interval.open(1, 3).intersects(Interval.closed(3, 4))

    assert Interval.point(3).intersects(Interval.closed(3, 4))
    assert Interval.point(3).intersects(Interval.closed(1, 3))
    assert not Interval.point(3).intersects(Interval.open(3, 4))
    assert not Interval.point(3).intersects(Interval.open(1, 3))

    assert Interval.closed(1, 3).intersects(Interval.closed(0, 1))
    assert not Interval.closed(1, 3).intersects(Interval.open(0, 1))
    assert not Interval.open(1, 3).intersects(Interval.closed(0, 1))

    assert not Interval.closed(1, 3).intersects(Interval.closed(4, 5))
    assert not Interval.closed(1, 3).intersects(Interval.closed(-2, 0))

    assert not Interval.closed(1, 3).intersects(Interval.empty())
    assert Interval.closed(1, 3).intersects(Interval.infinite())

    assert not Interval.point(1).intersects(Interval.open_closed(1, 2))

def test_parse():
    d = Interval.parse(Interval(0, 1, start_open=True, end_open=True))
    assert d.start == 0
    assert d.end == 1
    assert d.start_open
    assert d.end_open

    d = Interval.parse((0, 1))
    assert d.start == 0
    assert d.end == 1
    assert not d.start_open
    assert not d.end_open

    d = Interval.parse(1)
    assert d.start == 1
    assert d.end == 1
    assert not d.start_open
    assert not d.end_open

    with pytest.raises(Exception):
        _ = Interval.parse(None)
    with pytest.raises(Exception):
        _ = Interval.parse(None, default_inf=False)
    assert Interval.parse(None, default_inf=True) == Interval.infinite()

    d = Interval.parse(math.inf)
    assert math.isinf(d.start)
    assert math.isinf(d.end)
    assert d.start > 0
    assert d.end > 0
    assert not d.is_negative_infinite
    assert not d.is_positive_infinite

    d = Interval.parse(-math.inf)
    assert math.isinf(d.start)
    assert math.isinf(d.end)
    assert d.start < 0
    assert d.end < 0
    assert not d.is_negative_infinite
    assert not d.is_positive_infinite

    d = Interval.parse([])
    assert d.is_empty

def test_partition():
    ds = Interval(1, 3).partition([2])
    assert list(map(tuple, ds)) == [(1, 2), (2, 3)]
    assert not ds[0].start_open
    assert ds[0].end_open
    assert not ds[1].start_open
    assert not ds[1].end_open

    ds = Interval(0, 3).partition([0, 1, 2, 3, 4], start_open=True)
    assert list(map(tuple, ds)) == [(0, 0), (0, 1), (1, 2), (2, 3)]
    assert not ds[0].start_open
    assert not ds[0].end_open
    assert ds[1].start_open
    assert not ds[1].end_open

    ds = Interval(0, 3).partition([0, 1, 2, 3, 4], start_open=False)
    assert list(map(tuple, ds)) == [(0, 1), (1, 2), (2, 3), (3, 3)]
    assert not ds[0].start_open
    assert ds[0].end_open
    assert not ds[1].start_open
    assert ds[1].end_open

def test_subset():
    d = Interval(1, 3)
    assert d.is_subset_of((0, 4))
    assert d.is_subset_of((1, 3))
    assert not d.is_subset_of(Interval.closed_open(1, 3))

    assert d.is_superset_of((2, 2))
    assert d.is_superset_of((1, 3))
    assert d.is_superset_of(Interval.closed_open(1, 3))

def test_equals():
    d = Interval(1, 3)
    assert d.equals((1, 3))
    assert not d.equals(None)
    assert not d.equals(Interval.closed_open(1, 3))
    assert Interval.empty().equals(Interval.empty())
    # Empty intervals are always equal
    assert Interval.open(1, 1).equals(Interval.open(2, 2)) 
    assert Interval.infinite().equals(Interval.infinite())

def test_infinite():
    assert Interval.gte(math.inf).is_empty is True
    assert Interval.gte(-math.inf).is_empty is False
    assert Interval.lte(math.inf).is_empty is False
    assert Interval.lte(-math.inf).is_empty is True

def test_round():
    assert Interval(1.2, 3.4).round() == (1, 3)
    assert Interval(1.2, 3.4).round(method=math.floor) == (1, 3)
    assert Interval(1.2, 3.4).round(method=math.ceil) == (2, 4)
    assert Interval.open_closed(1.2, 3.4).round() == Interval.open_closed(1, 3)
    assert Interval.closed_open(1.2, 3.4).round() == Interval.closed_open(1, 3)
    assert Interval.empty().round() == Interval.empty()

def test_extensions():
    d = Interval(1, 3)
    assert d.get_lte().equals(Interval.lte(3))
    assert d.get_gte().equals(Interval.gte(1))
    assert d.get_lt().equals(Interval.lt(1))
    assert d.get_gt().equals(Interval.gt(3))

    d = Interval.open(1, 3)
    assert d.get_lte().equals(Interval.lt(3))
    assert d.get_gte().equals(Interval.gt(1))
    assert d.get_lt().equals(Interval.lte(1))
    assert d.get_gt().equals(Interval.gte(3))

    d = Interval.empty()
    assert d.get_lte().is_empty
    assert d.get_gte().is_empty
    assert d.get_lt().is_empty
    assert d.get_gt().is_empty

def test_inequalities():
    assert Interval(1, 3) == (1, 3)
    assert (1, 3) == Interval(1, 3)

    assert Interval(1, 3) < (4, 6)
    assert not Interval(1, 3) < (3, 6)
    assert not Interval(1, 3) < (-3, -1)
    assert Interval(1, 3) <= (3, 6)
    assert Interval(1, 3) <= (2, 6)
    assert Interval(1, 3) <= (1, 6)
    assert Interval(3, 5) <= (1, 6)
    assert not Interval(1, 3) <= (-3, -1)
    assert not Interval(3, 6) <= Interval.open(1, 6)

    assert Interval(1, 3) < Interval.empty()
    assert Interval(1, 3) <= Interval.empty()

    assert Interval(7, 9) > (4, 6)
    assert not Interval(7, 9) > (4, 7)
    assert not Interval(7, 9) > (10, 12)
    assert Interval(7, 9) >= (4, 7)
    assert Interval(7, 9) >= (4, 8)
    assert Interval(7, 9) >= (4, 9)
    assert not Interval(7, 9) >= (10, 12)
    assert not Interval(4, 10) >= Interval.open(4, 9)

    assert Interval(7, 9) > Interval.empty()
    assert Interval(7, 9) >= Interval.empty()

def test_arithmetic():
    assert Interval(1, 3) + (2, 4) == (1, 4)
    assert (1, 3) + Interval(2, 4) == (1, 4)

    assert Interval.open(1, 3) + (2, 4) == Interval.open_closed(1, 4)
    assert (1, 3) + Interval.open(2, 4) == Interval.closed_open(1, 4)
