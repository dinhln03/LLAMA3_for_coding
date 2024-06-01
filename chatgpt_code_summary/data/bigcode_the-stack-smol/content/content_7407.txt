#!/usr/bin/env python
import sys
from indicator import Indicator


def main():
    stationId = sys.argv[1] if len(sys.argv) > 1 else Indicator.DEFAULT_STATION_ID
    ind = Indicator(stationId)
    print(ind.get_aqindex_url())
    print(ind.get_all_stations_url())
    print(ind.get_aqindex())
    print(ind.get_data())
    return 0

if __name__ == '__main__':
    sys.exit(main())
