import pytest
from engine.constants import G
from pytest import param as p

from .orbit_derived_parameters import OrbitalPeriod


@pytest.mark.parametrize(
    ("primary_mass", "secondary_mass", "semimajor_axis", "expected"),
    [p(10e10, 100, 10, 76.9102, id="arbitrary period")],
)
def test_orbital_period(primary_mass, secondary_mass, semimajor_axis, expected):
    assert OrbitalPeriod(
        primary_mass, secondary_mass, semimajor_axis, G
    ).evalf() == pytest.approx(expected, 1e-3)
