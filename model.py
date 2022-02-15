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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from fpl_utils.parameters import (
    ele_types_cols, teams_cols, cut_cols, ele_cols, event_cols, team_list
)
from fpl_utils.fpl_functions import(
    split_bootstrap_data, get_historic_player_data, get_full_dataset,
    get_fixtures_data
)
from sklearn.model_selection import train_test_split
from smtplib import SMTP
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pretty_html_table import build_table

file_loc = r'/Users/timyouell/Documents/Coding/fantasy_football'

events_df = split_bootstrap_data('events', event_cols)
fixtures_df = get_fixtures_data()
elements_df = split_bootstrap_data('elements', ele_cols)
elements_types_df = split_bootstrap_data('element_types', ele_types_cols)
teams_df = split_bootstrap_data('teams', teams_cols)
hist_data = get_historic_player_data()
full_df = get_full_dataset(hist_data, cut_cols)


def get_upcoming_gw(df):
    cut_df = df[df['is_next'] == True].reset_index()
    gw = cut_df['name'].astype(str)[0]
    gw = gw.replace('Gameweek ', '')
    upcoming_gw = int(gw)
    if df['finished'][upcoming_gw-2] == False:
        df = full_df[(full_df['round'] < (upcoming_gw-1))]
        print('Gameweek ' + str(upcoming_gw-1) + ' still ongoing. ')
        print('Therefore, building model for Gameweek ' + str(upcoming_gw-1))
        return int(upcoming_gw - 1)
    else:
        return int(upcoming_gw)


#what is the gameweek coming up?
upcoming_gw = get_upcoming_gw(events_df)



def get_next5_fdr(upc_gw, teams_df):
    fixt_df = get_fixtures_data()
    fixt_df = fixt_df.loc[(fixt_df['event2'] >= upc_gw) & (fixt_df['event2'] <= upc_gw+4)]
    fixt_df['home_team_name'] = fixt_df['team_h'].map(teams_df.set_index('id')['short_name'])
    fixt_df['away_team_name'] = fixt_df['team_a'].map(teams_df.set_index('id')['short_name'])
    for team in team_list:
        if team == team_list[0]:
            first_df = fixt_df[(fixt_df['home_team_name'] == team) | (fixt_df['away_team_name'] == team)]
            first_df.sort_values(['event2', 'kickoff_time'], ascending=True, inplace=True)
            first_df['away_team_name'] = first_df['away_team_name'] + ' (H)'
            first_df['home_team_name'] = first_df['home_team_name'] + ' (A)'
            first_df['away_team_name'].replace(team + ' (H)', np.nan, inplace=True)
            first_df['home_team_name'].replace(team + ' (A)', np.nan, inplace=True)
            first_df['next5'] = first_df['home_team_name'].fillna(first_df['away_team_name'])
            first_df.loc[first_df['blank'] == True, 'next5'] = 'BLANK'
            first_df['fdr'] = first_df['next5'].str[:3].map(teams_df.set_index('short_name')['strength'])
            dup_df = first_df.duplicated(subset=['event2'], keep=False).reset_index()
            dup_df.columns = ['index', 'multiple']
            first_df = first_df.reset_index().merge(dup_df, on='index', how='left')
            first_df = first_df[~((first_df['multiple'] == True) & (first_df['blank'] == True))]
            first_df['next5_new'] = first_df.groupby(['event2'])['next5'] \
                .transform(lambda x : ' + '.join(x))
            fdr_gw_aves = first_df[['event2', 'fdr']].groupby('event2').mean().reset_index()
            fdr_gw_aves.columns = ['event2', 'fdr_gw_ave']
            first_df.drop_duplicates('event2', keep='first', inplace=True)
            first_df = first_df.merge(fdr_gw_aves, on='event2', how='left')
            sorted_df = pd.DataFrame(data={'event': np.arange(upc_gw, upc_gw+5)})
            sorted_df = sorted_df.merge(first_df, on='event', how='left')
            sorted_df['next5_new'].fillna('BLANK', inplace=True)
            new_df = pd.DataFrame(data={'short_name': [team],
                                        'GW' + str(upc_gw): [sorted_df['next5_new'][0]],
                                        'GW' + str(upc_gw+1): [sorted_df['next5_new'][1]],
                                        'GW' + str(upc_gw+2): [sorted_df['next5_new'][2]],
                                        'GW' + str(upc_gw+3): [sorted_df['next5_new'][3]],
                                        'GW' + str(upc_gw+4): [sorted_df['next5_new'][4]],
                                        'GW' + str(upc_gw) + '_fdr': [sorted_df['fdr_gw_ave'][0]],
                                        'GW' + str(upc_gw+1) + '_fdr': [sorted_df['fdr_gw_ave'][1]],
                                        'GW' + str(upc_gw+2) + '_fdr': [sorted_df['fdr_gw_ave'][2]],
                                        'GW' + str(upc_gw+3) + '_fdr': [sorted_df['fdr_gw_ave'][3]],
                                        'GW' + str(upc_gw+4) + '_fdr': [sorted_df['fdr_gw_ave'][4]]})
        else:
            rest_df = fixt_df[(fixt_df['home_team_name'] == team) | (fixt_df['away_team_name'] == team)]
            rest_df.sort_values(['event2', 'kickoff_time'], ascending=True, inplace=True)
            rest_df['away_team_name'] = rest_df['away_team_name'] + ' (H)'
            rest_df['home_team_name'] = rest_df['home_team_name'] + ' (A)'
            rest_df['away_team_name'].replace(team + ' (H)', np.nan, inplace=True)
            rest_df['home_team_name'].replace(team + ' (A)', np.nan, inplace=True)
            rest_df['next5'] = rest_df['home_team_name'].fillna(rest_df['away_team_name'])
            rest_df.loc[rest_df['blank'] == True, 'next5'] = 'BLANK'
            rest_df['fdr'] = rest_df['next5'].str[:3].map(teams_df.set_index('short_name')['strength'])
            dup_df = rest_df.duplicated(subset=['event2'], keep=False).reset_index()
            dup_df.columns = ['index', 'multiple']
            rest_df = rest_df.reset_index().merge(dup_df, on='index', how='left')
            rest_df = rest_df[~((rest_df['multiple'] == True) & (rest_df['blank'] == True))]
            rest_df['next5_new'] = rest_df.groupby(['event2'])['next5'] \
                .transform(lambda x : ' + '.join(x))
            rest_fdr_gw_aves = rest_df[['event2', 'fdr']].groupby('event2').mean().reset_index()
            rest_fdr_gw_aves.columns = ['event2', 'fdr_gw_ave']
            rest_df.drop_duplicates('event2', keep='first', inplace=True)
            rest_df = rest_df.merge(rest_fdr_gw_aves, on='event2', how='left')
            sorted_df2 = pd.DataFrame(data={'event': np.arange(upc_gw, upc_gw+5)})
            sorted_df2 = sorted_df2.merge(rest_df, on='event', how='left')
            sorted_df2['next5_new'].fillna('BLANK', inplace=True)
            two_df = pd.DataFrame(data={'short_name': [team],
                                        'GW' + str(upc_gw): [sorted_df2['next5_new'][0]],
                                        'GW' + str(upc_gw+1): [sorted_df2['next5_new'][1]],
                                        'GW' + str(upc_gw+2): [sorted_df2['next5_new'][2]],
                                        'GW' + str(upc_gw+3): [sorted_df2['next5_new'][3]],
                                        'GW' + str(upc_gw+4): [sorted_df2['next5_new'][4]],
                                        'GW' + str(upc_gw) + '_fdr': [sorted_df2['fdr_gw_ave'][0]],
                                        'GW' + str(upc_gw+1) + '_fdr': [sorted_df2['fdr_gw_ave'][1]],
                                        'GW' + str(upc_gw+2) + '_fdr': [sorted_df2['fdr_gw_ave'][2]],
                                        'GW' + str(upc_gw+3) + '_fdr': [sorted_df2['fdr_gw_ave'][3]],
                                        'GW' + str(upc_gw+4) + '_fdr': [sorted_df2['fdr_gw_ave'][4]]})
            new_df = new_df.append(two_df, ignore_index=True)
            fdr_list = ['GW'+str(num)+ '_fdr' for num in np.arange(upc_gw, upc_gw+5)]
            new_df['next5_fdr_ave'] = new_df[fdr_list].mean(axis=1)
            new_df.sort_values('next5_fdr_ave', ascending=True, inplace=True)
    return new_df


df_fixt = get_next5_fdr(upcoming_gw+1, teams_df)


# plot of fdrs with gw oppositions overlayed.
def plot_fdr_heatmap(df, upc_gw):
    gws = np.arange(upc_gw, upc_gw+5)
    df_fixt_data = df[['short_name'] + ['GW'+str(num) + '_fdr' for num in gws]]
    df_fixt_data.set_index('short_name', inplace=True)
    df_fixt_data.columns = ['GW' + str(num) for num in gws]
    df_annots = df[['short_name'] + ['GW'+str(num) for num in gws]]
    df_annots.set_index('short_name', inplace=True)
    fig, ax = plt.subplots()
    sns.heatmap(df_fixt_data, annot=df_annots, fmt = '', cmap='GnBu',
                linewidths=2.5, linecolor='pink', annot_kws={'size': 5})
    plt.show()

plot_fdr_heatmap(df_fixt, upcoming_gw+1)


# filter for upcoming gameweek and check to see if in the middle of a gameweek
gw_data = full_df[full_df['round'] == (upcoming_gw)]
df = full_df[(full_df['round'] < (upcoming_gw))]


'''
df = full_df[(full_df['total_minutes_FPGW'] > 89) &
             (full_df['round'] < gameweek)]
'''
train, test = train_test_split(df, test_size=0.3, shuffle=True)

# cut_at = int(0.8 * len(full_df['round'].value_counts()))
# train = full_df[(full_df['round'] <= cut_at)]
# test = full_df[full_df['round'] > cut_at]    


y_key = 'total_points'
nonmodel_vars = [y_key] + ['round', 'player', 'team_full', 'position',
                           'opponent_full', 'value', 'element', 'fixture']

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

#gw_data.sort_values('player_team_goals_FPGW', ascending=True, inplace=True)
#gw_data.drop_duplicates(['element', 'fixture'], keep='first', inplace=True)

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
gw['opponent_full'] = gw.groupby(['element'])['opponent_full'].transform(lambda x: ' + '.join(x))
gw['predicted_points'] = gw.groupby(['element'])['predicted_points'].transform(lambda x: x.sum())
gw.drop_duplicates('element', keep='first', inplace=True)
gw.sort_values('predicted_points', ascending=False, inplace=True)

# remove unavailable players
unav_df = elements_df[['id', 'news']]
unav_df.columns = ['element', 'news']
gw = gw.merge(unav_df, on='element', how='left')
gw = gw[gw['news'] == '']
gw.drop(['id', 'blank', 'news'], axis=1, inplace=True)

# create a team if money was no issue
def get_algo_team(df, loc, upcoming_gw):
    csv_nm = loc + '/gameweek_preds/' + 'GW_' + str(upcoming_gw) + '_team.csv'
    algo_team = []
    total_pred_points = 0
    pos_dict = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    team_dict = {'Arsenal': 3, 'Aston Villa': 3, 'Brentford': 3, 'Brighton': 3,
                 'Burnley': 3, 'Chelsea': 3, 'Crystal Palace': 3, 'Everton': 3,
                 'Leicester': 3, 'Leeds': 3, 'Liverpool': 3, 'Man City': 3,
                 'Man Utd': 3, 'Newcastle': 3, 'Norwich': 3, 'Southampton': 3,
                 'Spurs': 3, 'Watford': 3, 'West Ham': 3, 'Wolves': 3}
    for idx, row in df.iterrows():
        if row.element not in algo_team and pos_dict[row.position] != 0 and team_dict[row.team_full] != 0:
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
    df_keepers.loc[df_keepers[df_keepers.position=='GKP'].first_valid_index(), 'start'] = True
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
                                    (team_df_all.position=='MID')].last_valid_index(), 'start'] = False
        team_df_all.loc[team_df_all[(team_df_all.position=='FWD') &
                                    (team_df_all.start==False)].first_valid_index(), 'start'] = True
    elif start_def_count == 2:
        team_df_all.loc[team_df_all[(team_df_all.position=='MID') |
                                    (team_df_all.position=='FWD')].last_valid_index(), 'start'] = False
        team_df_all.loc[team_df_all[(team_df_all.position=='DEF') &
                                    (team_df_all.start==False)].first_valid_index(), 'start'] = True
    else:
        team_df_all = team_df_all
    print(team_df_all.drop(['element', 'opponent_full'], axis=1))
    print('Captain Pick: ' + str(captain_name))
    start_only = team_df_all[(team_df_all['start'] == True) &
                             (team_df_all['player'] != captain_name)]
    pred_total = sum(start_only['predicted_points']) + 2*captain_pred_points
    print('Total Predicted Points: ' + str(pred_total))
    team_df_all.to_csv(csv_nm, index=False)
    print('Gameweek ' + str(upcoming_gw) + ' Team saved to .csv')
    team_df_all.drop('element', axis=1, inplace=True)
    return team_df_all, captain_name


def get_top20_ppmillion(df, current_gw):
    full_df_current_gw = full_df[full_df['round'] == current_gw]
    full_df_current_gw.sort_values('total_total_points_FPGW', ascending=False, inplace=True)
    full_df_current_gw.drop_duplicates('element', keep='first', inplace=True)
    ppmil_df = full_df_current_gw[['player', 'team_full', 'position', 'value',
                                   'total_total_points_FPGW',
                                   'total_points_per_million_FPGW']]
    top20_ppmil_df = ppmil_df.sort_values('total_points_per_million_FPGW',
                                          ascending=False)[:20]
    return top20_ppmil_df
    

def get_top20_ppminute(df, current_gw):
    # points per minute table - must have played more than a quarter of all mins
    full_df_current_gw = df[df['round'] == current_gw]
    full_df_current_gw.sort_values('total_total_points_FPGW', ascending=False, inplace=True)
    full_df_current_gw.drop_duplicates('element', keep='first', inplace=True)
    ppmin_df = full_df_current_gw[['player', 'team_full', 'position',
                                   'total_minutes_FPGW',
                                   'total_total_points_FPGW',
                                   'total_points_per_minute_FPGW']]
    quart_total_mins = (current_gw * 90)/4
    ppmin_df_cut = ppmin_df[ppmin_df['total_minutes_FPGW'] >= quart_total_mins]
    top20_ppmin_df = ppmin_df_cut.sort_values('total_points_per_minute_FPGW',
                                              ascending=False)[:20]
    return top20_ppmin_df


def get_top20_diff(df, current_gw):
    # top differentials (points per selected %)
    # must have scored at least on average 2.5 points per gameweek?
    full_df_current_gw = df[df['round'] == current_gw]
    full_df_current_gw.sort_values('total_total_points_FPGW', ascending=False, inplace=True)
    full_df_current_gw.drop_duplicates('element', keep='first', inplace=True)
    sel_cut = elements_df[['id', 'selected_by_percent']]
    sel_df_new = full_df_current_gw.merge(sel_cut, left_on='element', right_on='id')
    sel_df = sel_df_new[['player', 'team_full', 'position',
                         'total_total_points_FPGW', 'selected_by_percent']]
    sel_df = sel_df[sel_df['total_total_points_FPGW'] >= current_gw * 2.5]
    sel_df['selected_by_percent'] = sel_df['selected_by_percent'].astype(float)
    sel_df['points_per_selected%'] = sel_df['total_total_points_FPGW']/sel_df['selected_by_percent']
    top20_diff_df = sel_df.sort_values('points_per_selected%', ascending=False)[:20]
    return top20_diff_df

algo_team_df, capt = get_algo_team(gw, file_loc, upcoming_gw)
current_gw = upcoming_gw
top20_ppmil = get_top20_ppmillion(full_df, current_gw)
top20_ppmin = get_top20_ppminute(full_df, current_gw)
top20_diff = get_top20_diff(full_df, current_gw)



message = MIMEMultipart()
message['Subject'] = "Tim\'s Top Tips - Gameweek " + str(upcoming_gw)
message['From'] = 'timpython3@gmail.com'
#email_list = ['timothy.youell@gmail.com', 'timmo2009@live.com']
#message['To'] = ', '.join(email_list)
message['To'] = 'timothy.youell@gmail.com'
intro_text = """\
<html>
  <body>
    <p>Hello and welcome to Tim's Top Tips.<br>
       Below you can find the algorithm predicted team of the week for the upcoming gameweek
       (no money constraints).<br> 
    </p>
  </body>
</html>
"""
ppmil_text = """\
<html>
  <body>
    <p>Below you can find the top 20 points per million (return on investment) players.<br>
    </p>
  </body>
</html>
"""
ppmin_text = """\
<html>
  <body>
    <p>Below you can find the top 20 points per minute players.<br>
    </p>
  </body>
</html>
"""
diff_text = """\
<html>
  <body>
    <p>Below you can find the top 20 differential players
    (must have scored more than an average of 2.5 points per gameweek to register.<br>
    </p>
  </body>
</html>
"""
outro_text = """\
<html>
  <body>
    <p>Author: Tim Youell
       <a href="https://twitter.com/TimYouell?t=C8Ei-gzG1g7AZidqCpsP5w&s=09">@TimYouell</a>  
    </p>
  </body>
</html>
"""
html_algo = algo_team_df.to_html(index=False)
ppmil_df = top20_ppmil.to_html(index=False)
ppmin_df = top20_ppmin.to_html(index=False)
diff_df = top20_diff.to_html(index=False)
intro_text += html_algo
ppmil_text += ppmil_df
ppmin_text += ppmin_df
diff_text += diff_df

message.attach(MIMEText(intro_text.encode('utf-8'),'html','utf-8'))
message.attach(MIMEText(ppmil_text.encode('utf-8'),'html','utf-8'))
message.attach(MIMEText(ppmin_text.encode('utf-8'),'html','utf-8'))
message.attach(MIMEText(diff_text.encode('utf-8'),'html','utf-8'))
msg_body = message.as_string()
server = SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(message['From'], 'Purchases28!')
server.sendmail(message['From'], message['To'], msg_body)
server.quit()
print("Mail sent successfully.")

