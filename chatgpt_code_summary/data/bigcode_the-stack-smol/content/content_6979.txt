#!/usr/bin/env python
# coding: utf-8

# # Exploring JHU COVID Case, Death, and Vaccine Information
# This notebook takes the live, updated data from JHU CSSE and GovEx, formats and simplifies it for my purposes, and saves it in csv files in the same directory. The two data sources use slightly different conventions and provide data for slightly different locations, so I standardized column names and kept only those rows common to both datasets. It makes most sense for this to be run once, so that the same data is used every time. In the future, it could be worthwhile to make the processes in this project run on 'live' data, but not for the purposes of this project at this time.
# 
# #### Data Sources
# * [Case Data - JHU CSSE](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv)
# * [Vaccine Data - JHU GovEx](https://raw.githubusercontent.com/govex/COVID-19/master/data_tables/vaccine_data/global_data/time_series_covid19_vaccine_doses_admin_global.csv)
# 
# #### Technical Sources
# * [Pandas Documentation](https://pandas.pydata.org/docs/)
# * [MatPlotLib.PyPlot Documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html)
# * [Standardizing Dates with `datetime.datetime` - Stack Overflow](https://stackoverflow.com/questions/4709652/python-regex-to-match-dates)
# * [Getting Only Date in `datetime.datetime`](https://stackoverflow.com/questions/18039680/django-get-only-date-from-datetime-strptime)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import sys


# ## Case Info

# In[2]:


case_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
print(case_data.shape)
case_data.head()


# 

# In[3]:


plt.scatter(case_data['3/23/20'], case_data['3/23/21'])
plt.xlim([0, 2000])
plt.ylim([0, 10000])
plt.title('Relations in COVID Case Count Over One Year in Different Countries')
plt.xlabel('Cases on 3/23/2020')
plt.ylabel('Cases on 3/23/2021')
plt.plot(range(2000))


# The above plot is pretty useless in terms of correlation since we know (logically) that total case numbers can only increase. However, it provides a good example of the extremity of difference in scale of typical case numbers (within the range plotted) between early 2020 and early 2021. I also used it just to make sure there wouldn't be any obvious weird things with the data.
# 
# The below table indicates mean case count for each day listed. The drastic change is obvious.

# In[4]:


case_data.mean(numeric_only=True)


# ## Vaccine Info

# In[5]:


vaccine_data = pd.read_csv('https://raw.githubusercontent.com/govex/COVID-19/master/data_tables/vaccine_data/global_data/time_series_covid19_vaccine_doses_admin_global.csv')
print(vaccine_data.shape)
vaccine_data.head()


# ## Standardizing Case and Vaccine Info

# The first step is to standardize columns by deleting unnecessary ones and establishing common naming conventions between the two files to minimize mistakes when referring to them:

# In[6]:


# Rename geographic columns in vaccine data to standardize
rename_conventions = {'Province_State': 'Province/State', 'Country_Region': 'Country', 'Country/Region': 'Country'}
case_data.rename(columns=rename_conventions, inplace=True)
vaccine_data.rename(columns=rename_conventions, inplace=True)

# Standardize dates 
import datetime
def date_fixer(old_date):
    data_type = ''
    is_date = False
    if len(old_date) == 10 and old_date[4] == '-': # is of format YYYY-MM-DD
        date = datetime.datetime.strptime(old_date,'%Y-%m-%d').date()
        data_type = 'Vaccinations'
        is_date = True
    elif len(old_date) >= 6 and old_date[2] == '/' or old_date[1] == '/': # is of format (M)M/(D)D/YY
        date = datetime.datetime.strptime(old_date, '%m/%d/%y').date()
        data_type = 'Cases'
        is_date = True
    return str('{}/{}/{} {}'.format(date.month, date.day, date.year, data_type)) if is_date else old_date + data_type

vaccine_data.rename(columns=date_fixer, inplace=True)
case_data.rename(columns=date_fixer, inplace=True)


# Next, I deleted the columns that weren't dates or Country/Region and State/Province. I may later want to use population, but not yet.

# In[7]:


case_data.drop(columns=['Lat', 'Long', 'Province/State'], inplace=True)
vaccine_data.drop(columns=['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Lat', 'Long_', 'Combined_Key', 'Population', 'Province/State'], inplace=True)


# Next, I sorted the data, filled in null values with 0, combined rows from the same country, and merged the dataframes.

# In[8]:


case_data.sort_values(by='Country', inplace=True)
vaccine_data.sort_values(by='Country', inplace=True)
vaccine_data.fillna(0.0, inplace=True)
case_data.fillna(0, inplace=True)
case_data = case_data.groupby(['Country']).sum()
vaccine_data = vaccine_data.groupby(['Country']).sum()
case_data.to_csv('case-data.csv')
vaccine_data.to_csv('vaccine-data.csv')
full_data = pd.merge(case_data, vaccine_data, how='inner', on='Country')
print('case data size:', case_data.shape, 'vaccine data size:', vaccine_data.shape, 'full data size:', full_data.shape)


# The next step was to look at all the country names, so I can manually see if I want to get rid of any. I decided to keep them all, at least for now.

# In[9]:


pd.set_option('display.max_seq_items', None)
full_data.index


# Finally, I saved the data into a csv file which can be referenced later. The below cell should really be run once only, so that the same data is used each time. One way to update this project could be to reload the data automatically.

# In[10]:


full_data.to_csv('full-data.csv')

