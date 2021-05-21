#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:17:05 2021

@author: timyouell
"""

import requests
import pandas as pd
import numpy as np
from parameters import (
    base_url, new_ele_cols, cols_list, cut_cols, team_stat_cols
)
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost.sklearn import XGBRegressor

def get_bootstrap_data(data_type):
    resp = requests.get(base_url + 'bootstrap-static/')
    if resp.status_code != 200:
        raise Exception('Response was status code ' + str(resp.status_code))
    data = resp.json()
    try:
        elements_data = pd.DataFrame(data[data_type])
        return elements_data
    except KeyError:
        print('Unable to reach bootstrap API successfully')


def split_bootstrap_data(data_type, columns):
    df = get_bootstrap_data(data_type)
    if isinstance(df, pd.DataFrame):
        df = df[columns]
    else:
        raise Exception('Error putting ' + data_type + ' data into table')
    return df
    

def get_fixtures_data():
    fixtures_url = base_url + '/fixtures/'
    resp = requests.get(fixtures_url)
    data = resp.json()
    fixtures_df = pd.DataFrame(data)
    return fixtures_df


def get_player_data(player_id):
    history_url = base_url + '/element-summary/' + str(player_id) + '/'
    resp = requests.get(history_url)
    hist_data = pd.DataFrame(resp.json()['history'])
    fixt_data = pd.DataFrame(resp.json()['fixtures'])
    fixt_data['element'] = player_id
    fixt_data.loc[fixt_data['is_home'] == True,
                  'opponent_team'] = fixt_data['team_a'].astype(float)
    fixt_data.loc[fixt_data['is_home'] == False,
                  'opponent_team'] = fixt_data['team_h'].astype(float)
    cols_to_change = {'id': 'fixture', 'event': 'round', 'is_home': 'was_home'}
    fixt_data.rename(columns=cols_to_change, inplace=True)
    fixt_data = fixt_data[['element', 'fixture', 'round','was_home',
                           'opponent_team']]
    concat_df = pd.concat([hist_data, fixt_data])
    concat = concat_df.reset_index().drop('index', axis=1)
    return concat


def get_player_ids(df):
    player_id_data = df[['id', 'web_name']]
    units = np.arange(len(player_id_data))
    ids = [player_id_data['id'][num] for num in units]
    names = [player_id_data['web_name'][num] for num in units] 
    player_ids = {}
    for i in range(0,len(ids)):
        player_ids[ids[i]] = names[i]
    return player_ids


def data_shifter(df, col):
    new_col = 'player_' + str(col)+ '_FPGW'
    df[new_col] = df[col].astype(float).shift(1)
    new_col = 'total_' + str(col)+ '_FPGW'
    df[new_col] = df[col].astype(float).cumsum().shift(1)
    return df


def get_team_stats(df):
    df.loc[df['was_home'] == True, 'team_goals'] = df['team_h_score']
    df.loc[df['was_home'] == False, 'team_goals'] = df['team_a_score']
    df.loc[df['was_home'] == False, 'team_conceded'] = df['team_h_score']
    df.loc[df['was_home'] == True, 'team_conceded'] = df['team_a_score']
    return df


def collate_player_history(df):
    ids = get_player_ids(df)
    for i, name in ids.items():
        print('Getting ' + str(name) + ' data')
        if i == 1:
            data = get_player_data(1)
            data = get_team_stats(data)
            for col in cols_list:
                data = data_shifter(data, col)
        else:
            new_data = get_player_data(i)
            new_data = get_team_stats(new_data)
            for col in cols_list:
                new_data = data_shifter(new_data, col)
            data = data.append(new_data, ignore_index=True)
    return data


def get_historic_player_data():
    elements_df = split_bootstrap_data('elements', ele_cols)
    elements_df.columns = new_ele_cols
    hist_data = collate_player_history(elements_df)
    hist_data['player'] = hist_data['element'].map(elements_df.set_index('id')['web_name'])
    hist_data['team'] = hist_data['element'].map(elements_df.set_index('id')['team'])
    hist_data['team_full'] = hist_data['team'].map(teams_df.set_index('id')['name'])
    hist_data['opponent_full'] = hist_data['opponent_team'].map(teams_df.set_index('id')['name'])
    hist_data['position_type'] = hist_data['element'].map(elements_df.set_index('id')['position_type'])
    hist_data['position'] = hist_data['position_type'].map(elements_types_df.set_index('id')['plural_name_short'])
    hist_data['player_value_FPGW'] = hist_data['value'].astype(float).shift(1)
    return hist_data


def create_team_df(df, col_add):
    new_df = df[['name'] + team_stat_cols]
    new_df.columns = [col_add + '_' + c for c in ['name'] + team_stat_cols]
    new_df.rename(columns={col_add + '_name': col_add + '_full'}, inplace=True)
    return new_df


def get_full_dataset(df, cols):
    hist_cut = df[cols]
    team_teams_df = create_team_df(teams_df, 'team')
    opponent_teams_df = create_team_df(teams_df, 'opponent')
    merged = hist_cut.merge(team_teams_df, on='team_full', how='left')
    merged = merged.merge(opponent_teams_df, on='opponent_full', how='left')
    home_away = pd.get_dummies(merged['was_home'])
    home_away.columns = ['away', 'home']
    team_full = pd.get_dummies(merged['team_full'])
    team_full.columns = ['team_' + col for col in team_full.columns.tolist()]
    oppo_full = pd.get_dummies(merged['opponent_full'])
    oppo_full.columns = ['oppo_' + col for col in oppo_full.columns.tolist()]
    position = pd.get_dummies(merged['position'])
    full = pd.concat([merged, home_away, team_full, oppo_full, position], axis=1)
    full.drop('was_home', axis=1, inplace=True)
    full['total_points_per_minute_FPGW'] = full['total_total_points_FPGW']/full['total_minutes_FPGW']
    full['total_points_per_million_FPGW'] = full['total_total_points_FPGW']/(full['player_value_FPGW']/10)
    return full