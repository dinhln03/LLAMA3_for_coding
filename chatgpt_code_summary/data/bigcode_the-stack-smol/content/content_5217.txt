# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Path of the file is stored in the variable path
data=pd.read_csv(path)
#Code starts here
data.rename(columns={'Total':'Total_Medals'},inplace=True)
# Data Loading 
data['Better_Event'] = np.where(data['Total_Summer']> data['Total_Winter'], 'Summer', 'Winter')
data['Better_Event'] =np.where(data['Total_Summer'] ==data['Total_Winter'],'Both',data['Better_Event'])
better_event=data['Better_Event'].value_counts().idxmax()

data.head()

# Summer or Winter


# Top 10

data.head(10)

# Plotting top 10


# Top Performing Countries
top_countries=data[['Country_Name','Total_Summer', 'Total_Winter','Total_Medals']]
top_countries=top_countries[:-1]
top_countries
# Best in the world 
def top_ten(df,col):
    country_list=[]
    country_list= list((top_countries.nlargest(10,col)['Country_Name']))
    return country_list
top_10_summer=top_ten(top_countries,'Total_Summer')
top_10_winter=top_ten(top_countries,'Total_Winter')
top_10=top_ten(top_countries,'Total_Medals')
a=set(top_10_summer).intersection(set(top_10_winter))
b=a.intersection(set(top_10))
common=list(b)
summer_df= data[data['Country_Name'].isin(top_10_summer)]
summer_df.head()
winter_df= data[data['Country_Name'].isin(top_10_winter)]
winter_df.head()
top_df= data[data['Country_Name'].isin(top_10)]
top_df.head()
plt.figure(figsize=(10,10))
plt.bar(summer_df['Country_Name'],summer_df['Total_Summer'])
plt.xticks(rotation=30)
plt.show()
summer_df['Golden_Ratio']=summer_df['Gold_Summer']/summer_df['Total_Summer']
summer_max_ratio=max(summer_df['Golden_Ratio'])
summer_max_ratio
summer_country_gold=summer_df.loc[summer_df['Gold_Summer'].idxmax(),'Country_Name']
summer_country_gold
winter_df['Golden_Ratio']=summer_df['Gold_Winter']/summer_df['Total_Winter']
winter_max_ratio=max(winter_df['Golden_Ratio'])
winter_country_gold=winter_df.loc[winter_df['Gold_Winter'].idxmax(),'Country_Name']
winter_country_gold
top_df['Golden_Ratio']=top_df['Gold_Total']/top_df['Total_Medals']
top_max_ratio=max(top_df['Golden_Ratio'])
top_country_gold=top_df.loc[top_df['Golden_Ratio'].idxmax(),'Country_Name']
top_country_gold
data_1=data[:-1]
data_1['Total_Points']=data_1['Gold_Total']*3+data_1['Silver_Total']*2+data_1['Bronze_Total']*1
most_points=max(data_1['Total_Points'])
most_points
best_country=data_1.loc[data_1['Total_Points'].idxmax(),'Country_Name']
best_country
# Plotting the best
best=data[data['Country_Name']==best_country]
best
best=best[['Gold_Total','Silver_Total','Bronze_Total']]
best
best.plot.bar()
plt.xlabel("United States")
plt.ylabel("Medals")
plt.xticks(rotation=45)




