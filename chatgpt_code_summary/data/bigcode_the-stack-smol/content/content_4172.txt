#CODE1---For preparing the list of DRUG side-effect relation from SIDER database---
#Python 3.6.5 |Anaconda, Inc.

import sys
import glob
import errno
import csv

path = '/home/16AT72P01/Excelra/SIDER1/output/adverse_effects.tsv'
files = glob.glob(path)

unique_sideeffect = set()
unique_drug = set()
unique_pair = set()

with open(path) as f1:
	reader = csv.DictReader(f1, quotechar='"', delimiter='\t', quoting=csv.QUOTE_ALL, skipinitialspace=True)
	print(reader)
	for row in reader:
		unique_drug.add(row['drug_name'])
		unique_sideeffect.add(row['adverse_effect'])
		val = row['drug_name']+"|"+row['adverse_effect']
		unique_pair.add(val)
f1.close()
print(len(unique_drug))
print(len(unique_sideeffect))
print(len(unique_pair))
			    	
