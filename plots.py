#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:14:24 2021

@author: timyouell
"""

"""
import sys
import os
os.chdir(r'/Users/timyouell/Documents/Coding/fantasy_football')
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fpl_utils.parameters import (
    ele_types_cols, teams_cols, cut_cols
)
from fpl_utils.fpl_functions import(
    split_bootstrap_data, get_historic_player_data, get_full_dataset
)


elements_types_df = split_bootstrap_data('element_types', ele_types_cols)
teams_df = split_bootstrap_data('teams', teams_cols)
hist_data = get_historic_player_data()
full_df = get_full_dataset(hist_data, cut_cols)


# 1) Histogram plot of total points
y_key = 'total_points'
sns.histplot(data=full_df, x=y_key, binwidth=1).set_title('FPL Points Density')

# 2) Histogram plot of total points with players who have played more than 90
full_df = full_df[full_df['total_minutes_FPGW'] > 89]
sns.histplot(data=full_df, x=y_key, binwidth=0.5).set_title('FPL Points Density')
plt.xticks(np.arange(int(full_df[y_key].min()), int(full_df[y_key].max()+1), 1))

# 3) Plot of value against total points to give an idea about most valuably players
full_df_sorted = full_df.sort_values('round', ascending=False)
full_df_sorted = full_df_sorted[full_df_sorted['total_points'].notnull()]
full_df_sorted.drop_duplicates('element', keep='first', inplace=True)
sns.scatterplot(full_df_sorted['value']/10, full_df_sorted['total_total_points_FPGW']).set_title('FPL Player ROI')

# 4) Top 20 points per million players
full_df_sorted['points_per_million'] = full_df_sorted['total_total_points_FPGW']/(full_df_sorted['value']/10)
full_df_sorted = full_df_sorted.sort_values('points_per_million', ascending=False)
full_df_sorted = full_df_sorted[['player', 'position', 'team_full',
                                 'total_minutes_FPGW', 'value',
                                 'total_total_points_FPGW',
                                 'points_per_million']]
full_df_top20 = full_df_sorted[:20]

# mention here about GKP's being great value for money
full_df_sorted.reset_index(inplace=True)
full_df_sorted.drop('index', axis=1, inplace=True)
full_df_sorted.index[full_df_sorted['player'] == 'Ederson'].tolist() #28
full_df_sorted.index[full_df_sorted['player'] == 'Alisson'].tolist() #57

# above reinforced by points per position table
pivot=full_df_sorted.pivot_table(index='position', values='points_per_million', aggfunc=np.mean).reset_index()
pp_position = pivot.sort_values('points_per_million',ascending=False)
# This season it looks like GKPs and Defenders hold the best 'value'

# 5) Top 20 points per minute players (players who have played more than 500 mins of football)
full_df_sorted['points_per_minute'] = full_df_sorted['total_total_points_FPGW']/(full_df_sorted['total_minutes_FPGW'])
full_df_sorted = full_df_sorted.sort_values('points_per_minute', ascending=False)
full_df_sorted = full_df_sorted[['player', 'position', 'team_full',
                                 'total_minutes_FPGW', 'value',
                                 'total_total_points_FPGW',
                                 'points_per_minute']]
full_df_mins = full_df_sorted[full_df_sorted['total_minutes_FPGW'] > 500][:20]
# Bale has been injured a lot of the season but his impact is huge when he
# does play. Scoring an FPL point on average every 10 minutes of play.
# What has been coined 'Pep Roullette', the constant changing of the Man City
# starting 11 has meant that players are well rested for Champions League
# nights and Premier League ties respectively. It's been a massive factor in
# City doing so well this season. However, it does make it very difficult to
# know which players are going to play in any given gameweek.
# (Foden/Gundogan/Mahrez/Torres) As you'd expect the big 'premium' options are
# pretty high up here (Fernandes/Kane/Salah/Son). Special mention to Jesse
# Lingard who having rarely made an appearance at Man Utd over the past few
# seasons has gone to West Ham and has been in amazing form, as proved by his
# points_per_minute score.


