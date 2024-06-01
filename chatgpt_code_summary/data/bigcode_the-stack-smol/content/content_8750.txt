# This script fetches Atelier 801 translation file and adds the required IDs into our own translation files

import sys
from urllib.request import urlopen
import zlib
from string import Template
import json

if len(sys.argv) < 2:
	print("Please pass in lang code for first arguement")
	exit()
lang = sys.argv[1]
url = 'https://www.transformice.com/langues/tfm-'+lang+'.gz'

# Fetch file
response = urlopen(url)
filedata = response.read()
filedata = zlib.decompress(filedata)
filedata = bytes.decode(filedata)

# Parse file
filedata = filedata.split("\n-\n")
i18n = {}
for data in filedata:
	if(not data): continue
	key,val = data.split("=", 1)
	i18n[key] = val

# Use data to do the actual thing this tool is for

def desc(key, arg1=None):
	if(arg1 != None):
		return i18n[key].replace("%1", arg1)
	return i18n[key]

transKeys = [
	"C_GuideSprirituel",
	"C_MaitresseDuVent",
	"C_Mecanicienne",
	"C_Sauvageonne",
	"C_Physicienne",

	"C_14", "C_14_T",
	"C_11", "C_11_T",
	"C_12", "C_12_T",
	"C_13", "C_13_T",
	"C_8", "C_8_T",
	"C_9", "C_9_T",
	"C_10", "C_10_T",
	"C_5", "C_5_T",
	"C_6", "C_6_T",
	"C_7", "C_7_T",
	"C_2", "C_2_T",
	"C_3", "C_3_T",
	"C_4", "C_4_T",
	"C_0", "C_0_T",
	"C_1", "C_1_T",
	"C_34", "C_34_T",
	"C_31", "C_31_T",
	"C_32", "C_32_T",
	"C_33", "C_33_T",
	"C_28", "C_28_T",
	"C_29", "C_29_T",
	"C_30", "C_30_T",
	"C_25", "C_25_T",
	"C_26", "C_26_T",
	"C_27", "C_27_T",
	"C_22", "C_22_T",
	"C_23", "C_23_T",
	"C_24", "C_24_T",
	"C_20", "C_20_T",
	"C_21", "C_21_T",
	"C_54", "C_54_T",
	"C_51", "C_51_T",
	"C_52", "C_52_T",
	"C_53", "C_53_T",
	"C_48", "C_48_T",
	"C_49", "C_49_T",
	"C_50", "C_50_T",
	"C_45", "C_45_T",
	"C_46", "C_46_T",
	"C_47", "C_47_T",
	"C_42", "C_42_T",
	"C_43", "C_43_T",
	"C_44", "C_44_T",
	"C_40", "C_40_T",
	"C_41", "C_41_T",
	"C_94", "C_94_T",
	"C_80", "C_80_T",
	"C_93", "C_93_T",
	"C_70", "C_70_T",
	"C_72", "C_72_T",
	"C_81", "C_81_T",
	"C_92", "C_92_T",
	"C_66", "C_66_T",
	"C_71", "C_71_T",
	"C_73", "C_73_T",
	"C_68", "C_68_T",
	"C_88", "C_88_T",
	"C_84", "C_84_T",
	"C_86", "C_86_T",
	"C_89", "C_89_T",
	"C_91", "C_91_T",
	"C_83", "C_83_T",
	"C_85", "C_85_T",
	"C_90", "C_90_T",
	"C_63", "C_63_T",
	"C_74", "C_74_T",
	"C_87", "C_87_T",
	"C_82", "C_82_T",
	"C_60", "C_60_T",
	"C_64", "C_64_T",
	"C_65", "C_65_T",
	"C_69", "C_69_T",
	"C_67", "C_67_T",
	"C_61", "C_61_T",
	"C_62", "C_62_T",
]

i18nToWrite = {}
for key in transKeys:
	i18nToWrite[key] = i18n[key]

with open(lang+'.json', 'w') as outfile:
	# outfile.write(i18nToWrite)
	json.dump(i18nToWrite, outfile, indent=4)