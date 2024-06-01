from Good_Boids_module.Update_Boids import Boids
import numpy as np
from nose.tools import assert_almost_equal, assert_greater
from nose.tools import assert_less, assert_equal
from numpy.testing import assert_array_equal
import os
import yaml
from Good_Boids_module.tests.record_fixtures import configuration_file


fixtures = yaml.load(open('fixture.yaml'))
configuration_file_data = yaml.load(open(configuration_file))


def test_good_boids_for_regression():

    before_positions = list(fixtures["before_positions"])
    before_velocities = list(fixtures["before_velocities"])

    new_positions = list(Boids(configuration_file).get_raw_positions(before_positions, before_velocities))
    after_positions = list(fixtures["after_positions"])

    new_velocities = list(Boids(configuration_file).get_raw_velocities(before_positions, before_velocities))
    after_velocities = list(fixtures["after_velocities"])

    for i in range(len(new_positions)):
        assert_almost_equal(new_positions[0][i], after_positions[0][i], delta=0.1)
        assert_almost_equal(new_positions[1][i], after_positions[1][i], delta=0.1)
        assert_almost_equal(new_velocities[0][i], after_velocities[0][i], delta=15)
        assert_almost_equal(new_velocities[1][i], after_velocities[1][i], delta=15)

test_good_boids_for_regression()


def test_good_boids_initialization():
    boids_positions = Boids(configuration_file).positions
    boids_velocities = Boids(configuration_file).velocities
    assert_equal(configuration_file_data['birds_number'], len(boids_positions[0]))
    assert_equal(configuration_file_data['birds_number'], Boids(configuration_file).birds_num)
    for boid in range(Boids(configuration_file).birds_num):
        assert_less(boids_positions[0][boid], configuration_file_data['position_upper_limits'][0])
        assert_greater(boids_positions[0][boid], configuration_file_data['position_lower_limits'][0])
        assert_less(boids_positions[1][boid], configuration_file_data['position_upper_limits'][1])
        assert_greater(boids_positions[1][boid], configuration_file_data['position_lower_limits'][1])
        assert_less(boids_velocities[0][boid], configuration_file_data['velocity_upper_limits'][0])
        assert_greater(boids_velocities[0][boid], configuration_file_data['velocity_lower_limits'][0])
        assert_less(boids_velocities[1][boid], configuration_file_data['velocity_upper_limits'][1])
        assert_greater(boids_velocities[1][boid], configuration_file_data['velocity_lower_limits'][1])

test_good_boids_initialization()