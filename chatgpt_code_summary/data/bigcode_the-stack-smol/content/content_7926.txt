# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:37:26 2021

@author: brian
"""
import os
os.chdir('C:/Users/brian/Desktop/All/UWEC/DS785_Capstone/Project')
import brawl_data as bd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

all_win_rates = bd.sql_get_results('dbname=BrawlStars user=postgres password=PG!3%7(', '', '', '', 0, 0, my_id = '', 
                   custom_query = "SELECT mode, map, brawler, wins, matches_played FROM population_aggs_high;")

all_win_rates['win_rate'] = all_win_rates['wins']/all_win_rates['matches_played']
all_win_rates = all_win_rates.loc[all_win_rates['matches_played']>10,:]
win_rate_extremes =  all_win_rates.groupby(['mode', 'map']).win_rate.agg(['min', 'max'])
win_rate_extremes = win_rate_extremes.reset_index()
win_rate_extremes['win_rate_differential'] = win_rate_extremes['max'] - win_rate_extremes['min']
win_rate_extremes = win_rate_extremes.sort_values(by = 'win_rate_differential')
win_rate_extremes.columns = ['Mode', 'Map', 'Minimum Brawler Win Rate', 'Maximum Brawler Win Rate', 'Win Rate Differential']
sns.set_style("darkgrid")
sns.scatterplot(data=win_rate_extremes, 
                x='Minimum Brawler Win Rate', 
                y='Maximum Brawler Win Rate', 
                hue='Win Rate Differential', 
                palette=sns.cubehelix_palette(start=2, rot=0, dark=.2, light=.8, as_cmap=True))
plt.title('Win Rates Differences for Brawlers Across Each Map-Mode')

sns.violinplot(x=win_rate_extremes['Win Rate Differential'])
plt.title('Differences Between Maximum and Minimum Win Rates for Brawlers Across Each Map-Mode')


for_example = all_win_rates.loc[all_win_rates['map'] == 'Split', :].sort_values('win_rate', ascending = False)
for_example = for_example.loc[:,['map', 'mode', 'brawler', 'win_rate']]
for_example = pd.concat([for_example.head(5),for_example.tail(5)])

for_example_2 = pd.concat([win_rate_extremes.head(5),win_rate_extremes.tail(5)])
for_example_2 = for_example_2.sort_values('Win Rate Differential', ascending=False)


example = bd.get_recommendation('dbname=BrawlStars user=postgres password=PG!3%7(', 'records', '#2G080980', 'brawlBall', 'Sneaky Fields', 0, 4)
example = pd.concat([example.head(5),example.tail(5)])

my_recs = bd.get_all_recommendations('dbname=BrawlStars user=postgres password=PG!3%7(', 'records', '#8VUPQ2PP', my_trophy_min = 500)

map_weaknesses = bd.get_map_weaknesses('dbname=BrawlStars user=postgres password=PG!3%7(', 'records')
map_weaknesses.head(10)

all_individual_history = bd.sql_get_results('dbname=BrawlStars user=postgres password=PG!3%7(', '', '', '', 0, 0, my_id = '', 
                   custom_query = "SELECT * FROM individual_aggs_high UNION ALL SELECT * FROM individual_aggs_mid UNION ALL SELECT * FROM individual_aggs_low;")
all_population_history = bd.sql_get_results('dbname=BrawlStars user=postgres password=PG!3%7(', '', '', '', 0, 0, my_id = '', 
                   custom_query = "SELECT * FROM population_aggs_high UNION ALL SELECT * FROM population_aggs_mid UNION ALL SELECT * FROM population_aggs_low;")

#Calculate win rate confidence intervals
all_individual_history['win_rate'] = all_individual_history['wins'] / all_individual_history['matches_played']
all_individual_history['ci.lower'],all_individual_history['ci.upper'] = zip(*all_individual_history.apply(lambda row : proportion_confint(count = row['wins'], nobs = row['matches_played'], alpha = .1, method = 'agresti_coull'), axis = 1))

all_population_history['win_rate'] = all_population_history['wins'] / all_population_history['matches_played']
all_individual_history = all_population_history.merge(all_individual_history, how = 'left', left_on = ['mode', 'map', 'brawler'], right_on = ['mode', 'map', 'brawler'])

#Compare population to individual history and inform recommendations
better = (all_individual_history['win_rate_x'] < all_individual_history['ci.lower']) & (all_individual_history['matches_played_y'] >= 5)
worse = (all_individual_history['win_rate_x'] > all_individual_history['ci.upper']) & (all_individual_history['matches_played_y'] >= 5)
sum(better) + sum(worse)