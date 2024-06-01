
import acequia as aq
from acequia import KnmiStations

if 1: # test KnmiStations.prec_stns()

    knmi = KnmiStations()
    
    print("Retrieving list of all available precipitation stations.")
    #filepath = r'..\02_data\knmi_locations\stn-prc-available.csv'
    filepath = 'stn-prc-available.csv'
    dfprc = knmi.prc_stns(filepath)
    print(f'{len(dfprc)} precipitation stations available on KMNI website.')
    print()

if 1: # test KnmiStations.wheater_stns()

    knmi = KnmiStations()
    print("retrieving list of all available weather stations.")
    #filepath = r'..\02_data\knmi_locations\stn-wht-available.csv'
    filepath = 'stn-wht-available.csv'
    dfwt = knmi.wtr_stns(filepath)
    print(f'{len(dfwt)} weather stations available on KMNI website.')
    print()


