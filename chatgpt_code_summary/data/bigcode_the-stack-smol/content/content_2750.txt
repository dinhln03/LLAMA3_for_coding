import pytest

from MidiCompose.logic.rhythm.beat import Beat
from MidiCompose.logic.rhythm.measure import Measure
from MidiCompose.logic.rhythm.part import Part

@pytest.fixture
def part_1():

    m1 = Measure([Beat([1,2,1,2]),
                  Beat([1,0,0,1])])
    m2 = Measure([Beat([2,2,1,1]),
                  Beat([2,2,2,2])])

    part = Part([m1,m2])

    return part

def test_empty_constructor():
    p = Part()
    assert p.n_measures == 1
    assert p.n_beats == 1
    assert p.n_note_on == 0

def test_n_note_on(part_1):

    assert part_1.n_note_on == 6


def test_iterator(part_1):
    for m in part_1:
        assert type(m) == Measure



