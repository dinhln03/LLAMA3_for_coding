from selenium import webdriver
#import itertools
from openpyxl import Workbook, load_workbook
import re
import datetime
driver = webdriver.Firefox()
driver.get("https://www.worldometers.info/coronavirus/")
countries = []
cases = []
newCases = []
data = []
casesInt = []
newCasesInt = []
cells = []
cellsB = []
datez = datetime.datetime.now()
nowDate = datez.strftime("%d%b%y")
for country in range(2,22):
	countries.append(driver.find_element_by_xpath("//table/tbody[1]/tr[" + str(country) + "]/td[1]").text)
for case in range(2,22):
	cases.append(driver.find_element_by_xpath("//table/tbody[1]/tr[" + str(case) + "]/td[2]").text)
for newCase in range(2,22):
	newCases.append(driver.find_element_by_xpath("//table/tbody[1]/tr[" + str(newCase) + "]/td[3]").text)
data = dict(zip(countries, zip(cases, newCases)))
#print(data)

for case in cases:
	case = re.sub(r'\D', '', case)
	casesInt.append(int(case))
for newCase in newCases:
	if newCase:
		newCase = re.sub(r'\D', '', newCase)
		newCasesInt.append(int(newCase))
	else:
		newCasesInt.append(1)

percentages = []
for caseInt,newCase in zip(casesInt, newCasesInt):
	result = caseInt - newCase
	percentage = round((newCase/result)*100, 2)
	percentages.append(percentage)
#for country, percentage in zip(countries, percentages):
#	print(country, ":", percentage)

wb = Workbook()
wb = load_workbook(filename='corona.xlsx')
ws = wb.active

#for countries column
for i in range(2,22):
	i = str(i)
	appendValue = 'A' + i
	appendValueB = 'B' + i
	cells.append(appendValue)
	cellsB.append(appendValueB)

for i in range(20):
	ws['A' + str(i+2)] = countries[i]
	ws['B' + str(i+2)] = percentages[i]

wb.save(filename="corona" + nowDate + ".xlsx")