#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 08:51:42 2022

@author: timyouell
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:18:47 2021

@author: timyouell
"""

from sklearn.metrics import r2_score, mean_absolute_error
from xgboost.sklearn import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from fpl_utils.parameters import (
    ele_types_cols, teams_cols, cut_cols, ele_cols, event_cols
)
from fpl_utils.fpl_functions import (
    split_bootstrap_data, get_historic_player_data, get_full_dataset,
    get_fixtures_data, get_upcoming_fixtures, get_upcoming_gw,
    plot_manager_history, get_top20_ppmillion, get_top20_ppminute,
    get_top20_diff
)
from sklearn.model_selection import train_test_split
from smtplib import SMTP
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

file_loc = r'/Users/timyouell/Documents/Python/FPL'

events_df = split_bootstrap_data('events', event_cols)
fixtures_df = get_fixtures_data()
elements_df = split_bootstrap_data('elements', ele_cols)
elements_types_df = split_bootstrap_data('element_types', ele_types_cols)
teams_df = split_bootstrap_data('teams', teams_cols)
hist_data = get_historic_player_data() # need to speed this up
full_df = get_full_dataset(hist_data, cut_cols)


# plot current form, my id is 35335
plot_manager_history(35335)

#what is the gameweek coming up?
upcoming_gw = get_upcoming_gw()
print('The upcoming Gameweek is GW' + str(upcoming_gw))

# View tables of stats so far
top20_ppmil = get_top20_ppmillion(full_df, upcoming_gw)
top20_ppmin = get_top20_ppminute(full_df, upcoming_gw)
top20_diff = get_top20_diff(full_df, upcoming_gw)

# get the upcoming fixtures for a certain number of gameweeks
df_fixt = get_upcoming_fixtures(upcoming_gw, teams_df, 6)

# filter for upcoming gameweek - train model on historic data
# gw_data = full_df[full_df['round'] == (upcoming_gw)]
df = full_df[(full_df['round'] < (upcoming_gw)) &
             (full_df['total_minutes_FPGW'] > 89)]

y_key = 'total_points'
nonmodel_vars = [y_key] + ['round', 'player', 'team_full', 'position',
                               'opponent_full', 'value', 'element', 'fixture']

def train_xgb_regr(df, y_key, nonmodel_vars):
    train, test = train_test_split(df, test_size=0.25, shuffle=True) 
    
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
    print('R2_Score: ' + str(r2_score(preds_df['actual'], preds_df['preds'])))
    print('MAE: ' + str(mean_absolute_error(preds_df['actual'], preds_df['preds'])))
    sns.regplot(preds_df['actual'], preds_df['preds'])
    plt.title('R^2 Score: ' + str(r2_score(preds_df['actual'], preds_df['preds'])))
    return xgb_regr


xgb_regr = train_xgb_regr(df, y_key, nonmodel_vars)


def get_specific_gw_preds(full_df, gw, model, y_key, nonmodel_vars):
    gw_data = full_df[full_df['round'] == gw]
    gw_dets = gw_data[nonmodel_vars]
    gw_ready = gw_data.drop(nonmodel_vars, axis=1)
    gw_preds = xgb_regr.predict(gw_ready)
    gw_preds = pd.DataFrame({'predicted_points': gw_preds})
    gw_dets = gw_dets.reset_index()
    gw_dets.drop('index', axis=1, inplace=True)
    gw_dets.reset_index(inplace=True)
    gw_preds.reset_index(inplace=True)
    gw = gw_dets.merge(gw_preds, on='index', how='left')
    gw.drop_duplicates(['element', 'fixture'], keep='first', inplace=True)
    gw.drop(['index', 'total_points', 'value'], axis=1, inplace=True)
    gw['predicted_points'] = round(gw['predicted_points'])
    gw['now_cost'] = gw['element'].map(elements_df.set_index('id')['now_cost'])
    blanks_df = fixtures_df[['id', 'blank']]
    gw = gw.merge(blanks_df, left_on='fixture', right_on='id')
    gw = gw[gw['blank'] == False]
    gw.drop('fixture', axis=1, inplace=True)
    gw['opponent_full'] = gw.groupby(['element'])['opponent_full'] \
        .transform(lambda x: ' + '.join(x))
    gw['predicted_points'] = gw.groupby(['element'])['predicted_points'] \
        .transform(lambda x: x.sum())
    gw.drop_duplicates('element', keep='first', inplace=True)
    gw.sort_values('predicted_points', ascending=False, inplace=True)
    unav_df = elements_df[['id', 'news']]
    unav_df.columns = ['element', 'news']
    gw = gw.merge(unav_df, on='element', how='left')
    gw = gw[gw['news'] == '']
    gw.drop(['id', 'blank', 'news'], axis=1, inplace=True)
    return gw


thirtythree = get_specific_gw_preds(full_df, 33, xgb_regr, y_key, nonmodel_vars)
thirtyfour = get_specific_gw_preds(full_df, 34, xgb_regr, y_key, nonmodel_vars)
thirtyfive = get_specific_gw_preds(full_df, 35, xgb_regr, y_key, nonmodel_vars)
thirtysix = get_specific_gw_preds(full_df, 36, xgb_regr, y_key, nonmodel_vars)
thityseven = get_specific_gw_preds(full_df, 37, xgb_regr, y_key, nonmodel_vars)
thirtyeight = get_specific_gw_preds(full_df, 38, xgb_regr, y_key, nonmodel_vars)




# create a team if money was no issue
def get_algo_team(df, loc, upcoming_gw):
    #csv_nm = loc + '/gameweek_preds/' + 'GW_' + str(upcoming_gw) + '_team.csv'
    algo_team = []
    total_pred_points = 0
    pos_dict = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    team_dict = {'Arsenal': 3, 'Aston Villa': 3, 'Brentford': 3, 'Brighton': 3,
                 'Burnley': 3, 'Chelsea': 3, 'Crystal Palace': 3, 'Everton': 3,
                 'Leicester': 3, 'Leeds': 3, 'Liverpool': 3, 'Man City': 3,
                 'Man Utd': 3, 'Newcastle': 3, 'Norwich': 3, 'Southampton': 3,
                 'Spurs': 3, 'Watford': 3, 'West Ham': 3, 'Wolves': 3}
    for idx, row in df.iterrows():
        if row.element not in algo_team and pos_dict[row.position] != 0 and \
            team_dict[row.team_full] != 0:
            algo_team.append(row.element)
            pos_dict[row.position] -= 1
            team_dict[row.team_full] -= 1
            total_pred_points += row.predicted_points
    team_df = pd.DataFrame(algo_team)
    new_df = df.drop(['round'], axis=1)
    team_df.columns = ['element']
    team_df = team_df.merge(new_df, on='element', how='left')
    captain_row = team_df['predicted_points'].idxmax()
    captain_name = team_df['player'][captain_row]
    captain_pred_points = team_df['predicted_points'][captain_row]
    team_df.loc[team_df['element'] >= 1, 'start'] = False
    # team_df['start'][team_df[team_df.position=='GKP'].first_valid_index()] = True
    df_no_keep = team_df[team_df['position'] != 'GKP']
    df_keepers = team_df[team_df['position'] == 'GKP']
    df_keepers.loc[df_keepers[df_keepers.position=='GKP'].first_valid_index(),
                   'start'] = True
    team_df_top10 = df_no_keep[:10]
    team_df_top10.loc[team_df_top10['element'] >= 1, 'start'] = True
    team_df_bench = df_no_keep[10:]
    team_df_all = pd.concat([df_keepers, team_df_top10, team_df_bench])
    # check to see if valid formation
    start_def_count = len(team_df_all[(team_df_all['start'] == True) &
                                      (team_df_all['position'] == 'DEF')])
    start_fwd_count = len(team_df_all[(team_df_all['start'] == True) &
                                      (team_df_all['position'] == 'FWD')])
    if start_fwd_count == 0:
        team_df_all.loc[team_df_all[(team_df_all.position=='DEF') |
                                    (team_df_all.position=='MID')] \
                        .last_valid_index(), 'start'] = False
        team_df_all.loc[team_df_all[(team_df_all.position=='FWD') &
                                    (team_df_all.start==False)] \
                        .first_valid_index(), 'start'] = True
    elif start_def_count == 2:
        team_df_all.loc[team_df_all[(team_df_all.position=='MID') |
                                    (team_df_all.position=='FWD')] \
                        .last_valid_index(), 'start'] = False
        team_df_all.loc[team_df_all[(team_df_all.position=='DEF') &
                                    (team_df_all.start==False)] \
                        .first_valid_index(), 'start'] = True
    else:
        team_df_all = team_df_all
    print(team_df_all.drop(['element', 'opponent_full'], axis=1))
    print('Captain Pick: ' + str(captain_name))
    start_only = team_df_all[(team_df_all['start'] == True) &
                             (team_df_all['player'] != captain_name)]
    pred_total = sum(start_only['predicted_points']) + 2*captain_pred_points
    print('Total Predicted Points: ' + str(pred_total))
    #team_df_all.to_csv(csv_nm, index=False)
    print('Gameweek ' + str(upcoming_gw) + ' Team saved to .csv')
    team_df_all.drop('element', axis=1, inplace=True)
    return team_df_all, captain_name

algo_team_df, capt = get_algo_team(thirtythree, file_loc, upcoming_gw)
