## 2. Introduction to the Data ##

import pandas as pd
all_ages = pd.read_csv('all-ages.csv')
recent_grads = pd.read_csv('recent-grads.csv')
print(all_ages.head())
print(recent_grads.head())

## 3. Summarizing Major Categories ##

# Unique values in Major_category column.
print(all_ages['Major_category'].unique())

aa_cat_counts = dict()
rg_cat_counts = dict()
def cat_summary(data,category):
    subset = data[data['Major_category']==category]
    total = subset['Total'].sum()
    return(total)
for cat in all_ages['Major_category'].unique():
    aa_cat_counts[cat] = cat_summary(all_ages,cat)
for cat in recent_grads['Major_category'].unique():
    rg_cat_counts[cat] = cat_summary(recent_grads,cat)
    

## 4. Low-Wage Job Rates ##

low_wage_percent = 0.0
low_wage_percent = recent_grads['Low_wage_jobs'].sum()/recent_grads['Total'].sum()

## 5. Comparing Data Sets ##

# All majors, common to both DataFrames
majors = recent_grads['Major'].unique()
rg_lower_count = 0
for item in majors:
    grad_subset = recent_grads[recent_grads['Major']==item]
    all_subset = all_ages[all_ages['Major']==item]
    if(grad_subset['Unemployment_rate'].values[0] < all_subset['Unemployment_rate'].values[0]):
        rg_lower_count +=1
print(rg_lower_count)        