from pandas import read_csv
from IPython.display import display
import numpy as np
import sys
import math

###############################
 ####Maria Eugenia Lopez ##### 
###############################


def fully_grown_depuration(number_to_remove=0.50):
	 return plants.loc[plants.height_m > number_to_remove]

def convert_GPS_lat_long(df):
	for index, row in df.iterrows():
		lat_viejo = row["GPS_lat"]
		latVal = (40008000*row["GPS_lat"])/360
		#res= div*0.001#to convert to Klm
		df.loc[index,"GPS_lat"] = latVal

		lat_radians = math.radians(lat_viejo)
		lonVal = (40075160*row["GPS_lon"])/360
		lonVal = lonVal*math.cos(lat_radians)
		#res = res*0.001
		df.loc[index,"GPS_lon"] = lonVal 

##----------------------------------------
##Part A Assembling a Data Set
##----------------------------------------

##----------------------------------------
##Input and Output: Data Frames

plants = read_csv('environmental_survey/plants2017.csv',
	index_col=0)

plants.reset_index(level=0,inplace=True)

plants.drop(plants.index[plants.Plant == 'tree'], inplace=True)
#display(plants.head(n=50))

plants.reset_index(drop=True,inplace=True)

##----------------------------------------
##Functions
convert_GPS_lat_long(	plants)
plants.rename(columns={'GPS_lon':'Meters_lon',
						'GPS_lat':'Meters_lat'}, inplace=True)


##----------------------------------------
##Functions and Data Structures: Boolean Indexing
heiht_set_by_user = float(input("Set the height that you want: ") or "0.5")
plants = fully_grown_depuration(float(heiht_set_by_user))

#reseting the index after the depuration 
plants.reset_index(drop=True,inplace=True)

display(plants)
