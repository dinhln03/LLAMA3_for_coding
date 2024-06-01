#!/usr/bin/python
# configure mini-split heat pumps for E files
# uses NRCan CSV list converted to TSV
# https://oee.nrcan.gc.ca/pml-lmp/index.cfm?language_langue=en&action=app.search-recherche&appliance=ASHP2_GH

import math, os, sys
import xml.etree.ElementTree as ET

if len(sys.argv) < 3:
    print(sys.argv[0], "E-file.h2k AHRI heads|0(ducted)")
    sys.exit()

e_file = sys.argv[1]
ahri = sys.argv[2]
heads = sys.argv[3]

t = ET.parse(e_file)

# tsv field list: 
# Brand	Outside model	Inside model	Furnace model	HSPF (Region IV)	Rated heating capacity (Btu/hour)	Grant amount	AHRI / Verification reference	AHRI Classification	Series name/product line (if applicable)	SEER	Rated cooling capacity (Btu/hour)	Coefficient of Performance (COP) at -15 °C (5 °F) (at maximum capacity)	Capacity Maintenance %  (Max -15°C/5°F ÷ Rated 8.3°C/47°F)
cchp_search = "grep '" + ahri + "' ccashp.tsv"
#d = os.popen(cchp_search).read().split('\t')
d = os.popen(cchp_search).read().rstrip('\n').split('\t')
# 1 kW = 3412 BTU/hr
(mfr, model, head_mdl, size_kw, hspf, seer, cop, fraction) = \
    d[0], d[1], d[2], str(float(d[5])/3412), d[4], d[10], d[12], d[13]
#(ahri, size_kw, hspf, cop, seer) = cols[9], str(float(cols[5])/3412), cols[4], cols[13], cols[12] 

e = t.find("./ProgramInformation/Information")

info = ET.Element("Info", {"code": "Info. 5"})
# no NEEP until spreadsheet and/or H2K is fixed
if (int(heads) > 0):
    info.text = "MSHP-" + heads
else:
    info.text = "CENTRAL-HP"
e.append(info)

# GHG instructions are to use Info 6 when more than 1 ccASHP system is installed
# but ENS wants all heat pumps in Info 6
info = ET.Element("Info", {"code": "Info. 6"})
info.text = mfr + ";AHRI-" + ahri + ';' + model + ';' + head_mdl 
e.append(info)

#print(info, info.attrib, info.text)

# Type 2 CCHP heating system
type2 = ET.parse("Type2.xml").getroot()
ahp = type2.find("AirHeatPump")
ei = ahp.find("EquipmentInformation")
ei.attrib["AHRI"] = ahri
ei.find("Manufacturer").text = mfr
ei.find("Model").text = model
ahp.find("Equipment").attrib["numberOfHeads"] = heads
specs = ahp.find("Specifications")
specs.find("OutputCapacity").attrib["value"] = size_kw
# use ASHP HSPF/SEER until NEEP spreadsheet or H2K is fixed for ccHP
specs.find("HeatingEfficiency").attrib["value"] = str(float(hspf)/1.15)
specs.find("CoolingEfficiency").attrib["value"] = seer
cchp = ahp.find("ColdClimateHeatPump")
cchp.attrib["heatingEfficiency"] = hspf
cchp.attrib["coolingEfficiency"] = seer
cchp.attrib["capacity"] = size_kw
cchp.attrib["cop"] = cop
cchp.attrib["capacityMaintenance"] = fraction

hc = t.find("./House/HeatingCooling")
hc.remove(hc.find("Type2"))
hc.append(type2)

#outfile = "MSHP-out.h2k"
outfile = e_file
t.write(outfile, "UTF-8", True)

