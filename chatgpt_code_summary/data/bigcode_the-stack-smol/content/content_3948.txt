
from .station import consistant_typical_range_stations


def stations_level_over_threshold(stations: list, tol: float) -> list:
    """function takes in stations and returns a list of tuples contating station and
    relative water lever where the relative water level greater than tol """

    stations = consistant_typical_range_stations(stations)  # gets consistant stations

    res_list = []

    for station in stations:

        rel_level = station.relative_water_level()

        if rel_level is not None:  # ensures water level is not None
            if rel_level > tol:
                res_list.append((station, rel_level))

    return res_list


def stations_highest_rel_level(stations, N):
    """Returns a list of N MonitoringStation objects ordered from highest to lowest risk"""
    stations = consistant_typical_range_stations(stations)

    def key(x):
        if x.relative_water_level() is not None:
            return x.relative_water_level()
        else:
            return float(0)
    stationByHighestLevel = sorted(stations, key=key, reverse=True)  # Hoping this will work we shall see
    NstationByLevel = stationByHighestLevel[:N]
    return NstationByLevel
