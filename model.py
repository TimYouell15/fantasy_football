#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:18:47 2021

@author: timyouell
"""

"""
import sys
import os
os.chdir(r'/Users/timyouell/Documents/Coding/fantasy_football')
"""

from sklearn.metrics import r2_score, mean_absolute_error
from xgboost.sklearn import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from fpl_utils.parameters import (
    ele_types_cols, teams_cols, cut_cols, ele_cols
)
from fpl_utils.fpl_functions import(
    split_bootstrap_data, get_historic_player_data, get_full_dataset
)
from sklearn.model_selection import train_test_split

elements_df = split_bootstrap_data('elements', ele_cols)
elements_types_df = split_bootstrap_data('element_types', ele_types_cols)
teams_df = split_bootstrap_data('teams', teams_cols)
hist_data = get_historic_player_data()
full_df = get_full_dataset(hist_data, cut_cols)


gw38_data = full_df[full_df['round'] == 38]

df = full_df[(full_df['total_minutes_FPGW'] > 89) &
             (full_df['round'] < 38)]

train, test = train_test_split(df, test_size=0.3, shuffle=True)

# cut_at = int(0.8 * len(full_df['round'].value_counts()))
# train = full_df[(full_df['round'] <= cut_at)]
# test = full_df[full_df['round'] > cut_at]


y_key = 'total_points'
nonmodel_vars = [y_key] + ['round', 'player', 'team_full', 'position',
                           'opponent_full', 'value', 'element']

train_out = train[nonmodel_vars]
test_out = test[nonmodel_vars]

train.drop(nonmodel_vars, axis=1, inplace=True)
test.drop(nonmodel_vars, axis=1, inplace=True)


X_train = train
Y_train = train_out[y_key]

X_test = test
Y_test = test_out[y_key]


xgb_regr = XGBRegressor(n_estimators=500,
                        learning_rate=0.01,
                        max_depth=5,
                        min_samples_leaf=5)

xgb_regr.fit(X_train, Y_train)



preds = xgb_regr.predict(X_test)
preds_df = pd.DataFrame({'actual': Y_test, 'preds': preds})
preds_df['preds'] = round(preds_df['preds'])

print(r2_score(preds_df['actual'], preds_df['preds']))
print(mean_absolute_error(preds_df['actual'], preds_df['preds']))


sns.regplot(preds_df['actual'], preds_df['preds'])
plt.title('R^2 Score: ' + str(r2_score(preds_df['actual'], preds_df['preds'])))

gw_38_dets = gw38_data[nonmodel_vars]
gw_38_ready = gw38_data.drop(nonmodel_vars, axis=1)
gw_38_preds = xgb_regr.predict(gw_38_ready)

gw38_preds = pd.DataFrame({'predicted_points': gw_38_preds})



# concat = pd.concat([gw_38_dets, gw38_preds], axis=1)

gw_38_dets = gw_38_dets.reset_index()
gw_38_dets.drop('index', axis=1, inplace=True)

gw_38_dets.reset_index(inplace=True)
gw38_preds.reset_index(inplace=True)

gw38 = gw_38_dets.merge(gw38_preds, on='index', how='left')
gw38.drop(['index', 'total_points', 'value'], axis=1, inplace=True)
gw38['predicted_points'] = round(gw38['predicted_points'])


gw38['now_cost'] = gw38['element'].map(elements_df.set_index('id')['now_cost'])
gw38.sort_values('predicted_points', ascending=False, inplace=True)


'''
GK = Ramsdale    (SHU) £4.6  pp = 5 Start
GK = Areola      (FUL) £4.5  pp = 4 Bench

DEF = Tarkowski  (BUR) £5.3  pp = 6 Start
DEF = Alioski    (LEE) £4.3  pp = 4 Start
DEF = Dallas     (LEE) £5.4  pp = 4 Start
DEF = Targett    (AVL) £5.0  pp = 4 Bench
DEF = Coufal     (WHU) £4.8  pp = 4 Bench

MID = Westwood   (BUR) £5.3  pp = 9 Start
MID = Salah      (LIV) £12.8 pp = 7 Start
MID = McNeil     (BUR) £5.7  pp = 6 Start
MID = Pereira    (WBA) £5.4  pp = 5 Start
MID = Højbjerg   (TOT) £4.9  pp = 5 Bench

FWD = Adams      (SOU) £5.7  pp = 6 Start
FWD = Kane       (TOT) £11.9 pp = 5 Start
FWD = Rodrigo    (LEE) £5.7  pp = 5 Start

Total Predicted Points = 62
Total Team Cost        = £91.3
'''

# work to be done: build rules set that automatically sorts the best 11, captain and team

def get_unavailable_players_list(df):
    df_cut = df[(df['status'] == 'u') | (df['status'] == 'd') | (df['status'] == 'i')]
    unav_list = df_cut['web_name'].tolist()
    return unav_list


def get_algo_team():
    algo_team = []
    sp_limit = 3
    budget = 1000
    unav_list = get_unavailable_players_list(elements_df)
    pos_dict = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    return algo_team
    
algo_team = get_algo_team()
