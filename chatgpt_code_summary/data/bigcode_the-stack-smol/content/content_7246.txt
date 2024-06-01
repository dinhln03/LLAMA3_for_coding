import random

# create the initial array
regionsEMEA = ["Central Eastern Europe", "France", "Germany", "Middle East / Africa", "United Kingdom", "Western Europe"]

# randomly pick region after region
num = len(regionsEMEA)
for x in range(num):
    numRegions = len(regionsEMEA)
    pos = random.randint(0,numRegions-1)
    selected = regionsEMEA[pos]
    print(selected)
    regionsEMEA.pop(pos)
