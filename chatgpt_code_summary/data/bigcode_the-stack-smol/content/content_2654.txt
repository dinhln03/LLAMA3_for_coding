"""Tests for functions defined in the floodsystem/geo module
"""

from floodsystem import geo
from floodsystem.station import MonitoringStation
from floodsystem.stationdata import build_station_list

stations = build_station_list()

# define arbitrary stations for the tests

station_id1 = "test station id 1"
measure_id1 = "test measure id 1"
label1 = "TS1"
coord1 = (1.0, 4.0)
typical_range1 = (-2, 5)
river1 = "River Cam"
town1 = "Town 1"
TestStation1 = MonitoringStation(station_id1, measure_id1, label1, coord1, typical_range1, river1, town1)

station_id2 = "test station id 2"
measure_id2 = "test measure id 2"
label2 = "TS2"
coord2 = (0.0, 1.0)
typical_range2 = (-2, 2)
river2 = "River Cam"
town2 = "Town 2"
TestStation2 = MonitoringStation(station_id2, measure_id2, label2, coord2, typical_range2, river2, town2)

station_id3 = "test station id 3"
measure_id3 = "test measure id 3"
label3 = "TS3"
coord3 = (1.0, 1.0)
typical_range3 = (-2, 3)
river3 = "River Thames"
town3 = "Town 3"
TestStation3 = MonitoringStation(station_id3, measure_id3, label3, coord3, typical_range3, river3, town3)

test_stations = [TestStation1, TestStation2, TestStation3]


def test_stations_within_radius():
    centre = (52.2053, 0.1218)

    # check that no stations are at a negative distance from the centre
    assert geo.stations_within_radius(stations, centre, 0) == []
    # check that all stations are within 10000km of the centre
    assert len(geo.stations_within_radius(stations, centre, 10000)) == len(stations)

def test_rivers_by_station_number():
    lst = geo.rivers_by_station_number(stations, 2)

    # check that the number of stations is greater (or equal to the second one) for the first river.
    assert lst[0][1] >= lst[1][1]

def test_stations_by_distance():
    test = geo.stations_by_distance(test_stations, (0,0))

    # check that the results are in the right order based on the test stations provided above
    assert (test[0][0], test[1][0], test[2][0]) == (TestStation2, TestStation3, TestStation1)

def test_rivers_with_station():

    # check that the results are River Cam and River Thames as per the test stations provided above
    assert geo.rivers_with_station(test_stations) == ['River Cam', 'River Thames']

def test_stations_by_river():

    # check that the two stations on the River Cam are TestStation1 and TestStation2
    assert sorted([x.name for x in geo.stations_by_river(test_stations)['River Cam']]) == [TestStation1.name, TestStation2.name]
