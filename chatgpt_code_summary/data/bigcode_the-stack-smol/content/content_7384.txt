import datetime
from datetime import timedelta
import csv
from module.stock import Stock, Market

# Historic Files
DAX = 'DAX.csv'
DOW = 'Dow.csv'
FTSE = 'FTSE.csv'

# Output File
OUTPUT = 'output.txt'

def import_csv(file_name):
    data = []
    with open(file_name) as file:
        reader = csv.reader(file)
        data = list(reader)
        return(data)

# Import historic stock performance
Dax_data = import_csv(DAX)
Dow_data = import_csv(DOW)
Ftse_data = import_csv(FTSE)

Ftse = Market('FTSE', Ftse_data)
Ftse.p.sort(key=lambda x: x.d, reverse=False)    # Sorts into date order
vanguard = Stock('Vanguard', 1, 100000)  # Name, price, units
vanguard.sim(Ftse, datetime.date(2007, 1, 1), datetime.date(2012, 1, 1))
print(vanguard.v)

''' Simulation '''
'''for i in range(0,1000):
    Ftse = Market('Random', [])
    Ftse.p.sort(key=lambda x: x.d, reverse=False)
    vanguard = Stock('Vanguard', 1, 100000)  # Name, price, units
    vanguard.sim(Ftse, datetime.date(2019, 2, 1), datetime.date(2022, 1, 1))
    print(vanguard.v)
    if(vanguard.v > 100000):
        break
'''
