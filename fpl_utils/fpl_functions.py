# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fpl_utils.parameters import (
    base_url, new_ele_cols, cols_list, team_stat_cols, ele_cols,
    teams_cols, ele_types_cols, event_cols
)

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
 

def get_manager_history_data(manager_id):
    global base_url
    manager_hist_url = base_url + 'entry/' + str(manager_id) + '/history/'
    resp = requests.get(manager_hist_url)
    if resp.status_code != 200:
        raise Exception('Response was status code ' + str(resp.status_code))
    json = resp.json()
    try:
        data = pd.DataFrame(json['current'])
        return data
    except KeyError:
        print('Unable to reach bootstrap API successfully')


def get_manager_info_json(manager_id):
    global base_url
    manager_hist_url = base_url + 'entry/' + str(manager_id) + '/'
    resp = requests.get(manager_hist_url)
    if resp.status_code != 200:
        raise Exception('Response was status code ' + str(resp.status_code))
    json = resp.json()
    return json


def plot_manager_history(player_id):
    man_hist_df = get_manager_history_data(player_id)
    man_info = get_manager_info_json(player_id)
    title = man_info['player_first_name'] + " " + \
            man_info['player_last_name'] + "'s " + \
            man_info['name'] + " Season Form"
    x = man_hist_df['event']
    y = man_hist_df['overall_rank']
    plt.plot(x, y)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel('Gameweek')
    plt.ylabel('Overall Rank')
    plt.ticklabel_format(style='plain')
    plt.show()


def split_bootstrap_data(data_type, columns):
    df = get_bootstrap_data(data_type)
    if isinstance(df, pd.DataFrame):
        df = df[columns]
        if data_type == 'elements':
            df = df[df['status'] != 'u']
        else:
            df = df
    else:
        raise Exception('Error putting ' + data_type + ' data into table')
    return df
    

def get_fixtures_data():
    fixtures_url = base_url + '/fixtures/'
    resp = requests.get(fixtures_url)
    data = resp.json()
    fixtures_df = pd.DataFrame(data)
    gw_dict = dict(zip(np.arange(1, 381),
                       [num for num in np.arange(1, 39) for x in range(10)]))
    fixtures_df.loc[fixtures_df['event'].isnull(),
                    'event2'] = fixtures_df['id'].map(gw_dict)
    fixtures_df['event2'].fillna(fixtures_df['event'], inplace=True)
    fixtures_df.loc[fixtures_df['event'].isnull(), 'blank'] = True
    fixtures_df['blank'].fillna(False, inplace=True)
    fixtures_df.sort_values('event2', ascending=True, inplace=True)
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
    #units = np.arange(len(player_id_data))
    units = np.array(player_id_data.index)
    ids = [player_id_data['id'][num] for num in units]
    names = [player_id_data['web_name'][num] for num in units] 
    player_ids = {}
    for i in range(0,len(ids)):
        player_ids[ids[i]] = names[i]
    return player_ids


def data_shifter(df, col):
    upc_gw = get_upcoming_gw()
    new_col = 'player_' + str(col) + '_FPGW'
    df[new_col] = df[col].astype(float).shift(1)
    new_col = 'total_' + str(col) + '_FPGW'
    df[new_col] = df[col].astype(float).cumsum().shift(1)
    ele_old = df[df['round'] < upc_gw]
    ele_future = df[df['round'] >= upc_gw]
    ele_most_recent = df[df['round'] == upc_gw-1]
    ele_non_fpg = df[[col for col in df.columns if '_FPGW' not in col]]
    ele_fpg = ele_most_recent[[col for col in df.columns if '_FPGW' in col]]
    ele_future = ele_non_fpg[ele_non_fpg['round'] >= upc_gw].reset_index()
    ele_new = pd.DataFrame(np.repeat(ele_fpg.values, len(ele_future),
                                     axis=0), columns=ele_fpg.columns)
    ele_new_ov = ele_future.join(ele_new).set_index('index')
    ele_total = pd.concat([ele_old, ele_new_ov])
    return ele_total


def get_team_stats(df):
    df.loc[df['was_home'] == True, 'team_goals'] = df['team_h_score']
    df.loc[df['was_home'] == False, 'team_goals'] = df['team_a_score']
    df.loc[df['was_home'] == False, 'team_conceded'] = df['team_h_score']
    df.loc[df['was_home'] == True, 'team_conceded'] = df['team_a_score']
    return df


def collate_player_history(df):
    form_df = df[df['form'].astype(str) != '0.0']
    ids = get_player_ids(form_df)
    for i, name in ids.items():
        print('Getting ' + str(name) + ' data')
        if i == int(list(ids.keys())[0]):
            data = get_player_data(int(list(ids.keys())[0]))
            data = get_team_stats(data)
            for col in cols_list:
                data = data_shifter(data, col)
        else:
            new_data = get_player_data(i)
            try:
                new_data = get_team_stats(new_data)
            except KeyError:
                print('Not all new player data loaded yet - try again later')
                break
            for col in cols_list:
                new_data = data_shifter(new_data, col)
            data = data.append(new_data, ignore_index=True)
    return data


def get_historic_player_data():
    elements_df = split_bootstrap_data('elements', ele_cols)
    elements_df.columns = new_ele_cols
    hist_data = collate_player_history(elements_df)
    teams_df = split_bootstrap_data('teams', teams_cols)
    elements_types_df = split_bootstrap_data('element_types', ele_types_cols)
    hist_data['player'] = hist_data['element'] \
        .map(elements_df.set_index('id')['web_name'])
    hist_data['team'] = hist_data['element'] \
        .map(elements_df.set_index('id')['team'])
    hist_data['team_full'] = hist_data['team'] \
        .map(teams_df.set_index('id')['name'])
    hist_data['opponent_full'] = hist_data['opponent_team'] \
        .map(teams_df.set_index('id')['name'])
    hist_data['position_type'] = hist_data['element'] \
        .map(elements_df.set_index('id')['position_type'])
    hist_data['position'] = hist_data['position_type'] \
        .map(elements_types_df.set_index('id')['plural_name_short'])
    hist_data['player_value_FPGW'] = hist_data['value'].astype(float).shift(1)
    return hist_data


def create_team_df(df, col_add):
    new_df = df[['name'] + team_stat_cols]
    new_df.columns = [col_add + '_' + c for c in ['name'] + team_stat_cols]
    new_df.rename(columns={col_add + '_name': col_add + '_full'}, inplace=True)
    return new_df


def get_full_dataset(df, cols):
    hist_cut = df[cols]
    teams_df = split_bootstrap_data('teams', teams_cols)
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
    full = pd.concat([merged, home_away, team_full, oppo_full, position],
                     axis=1)
    full.drop('was_home', axis=1, inplace=True)
    full['total_points_per_minute_FPGW'] = full['total_total_points_FPGW']/ \
        full['total_minutes_FPGW']
    full['total_points_per_million_FPGW'] = full['total_total_points_FPGW']/ \
        (full['player_value_FPGW']/10)
    return full


def get_upcoming_gw():
    df = split_bootstrap_data('events', event_cols)
    cut_df = df[df['is_next'] == True].reset_index()
    gw = cut_df['name'].astype(str)[0]
    gw = gw.replace('Gameweek ', '')
    upcoming_gw = int(gw)
    #print('The upcoming Gameweek is GW' + str(upcoming_gw))
    return int(upcoming_gw)


def get_new_df_from_sorted(upc_gw, sorted_df, team, fixt_num):
    gw_array = np.arange(fixt_num)
    gw_cols = ['GW' + str(upc_gw+num) for num in gw_array]
    fixt_next = [sorted_df['next5_new'][num] for num in gw_array]
    gw_fdr_cols = ['GW' + str(upc_gw+num) + '_fdr' for num in gw_array]
    fixt_fdr_next = [sorted_df['fdr_gw_ave'][num] for num in gw_array]
    cols = ['short_name'] + gw_cols + gw_fdr_cols
    fixt_data = [team] + fixt_next + fixt_fdr_next
    new_df = pd.DataFrame([fixt_data], columns=cols)
    return new_df


def plot_fdr_heatmap(df, upc_gw, fixt_num):
    gws = np.arange(upc_gw, upc_gw+fixt_num)
    df_fixt_data = df[['short_name'] + ['GW'+str(num) + '_fdr' for num in gws]]
    df_fixt_data.set_index('short_name', inplace=True)
    df_fixt_data.columns = ['GW' + str(num) for num in gws]
    df_annots = df[['short_name'] + ['GW'+str(num) for num in gws]]
    df_annots.set_index('short_name', inplace=True)
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5, forward=True)
    sns.heatmap(df_fixt_data, annot=df_annots, fmt = '', cmap='GnBu',
                linewidths=2.5, linecolor='pink', annot_kws={'size': 10})
    plt.show()


def get_team_list(teams_cols):
    teams_df = split_bootstrap_data('teams', teams_cols)
    team_list = teams_df['short_name'].tolist()
    return team_list


def get_upcoming_fixtures(upc_gw, teams_df, fixt_num):
    team_list = get_team_list(teams_cols)
    fixt_df = get_fixtures_data()
    fixt_df = fixt_df.loc[(fixt_df['event2'] >= upc_gw) &
                          (fixt_df['event2'] < upc_gw + fixt_num)]
    fixt_df['home_team_name'] = fixt_df['team_h'] \
        .map(teams_df.set_index('id')['short_name'])
    fixt_df['away_team_name'] = fixt_df['team_a'] \
        .map(teams_df.set_index('id')['short_name'])
    for team in team_list:
        if team == team_list[0]:
            first_df = fixt_df[(fixt_df['home_team_name'] == team) |
                               (fixt_df['away_team_name'] == team)]
            first_df.sort_values(['event2', 'kickoff_time'],
                                 ascending=True, inplace=True)
            first_df['away_team_name'] = first_df['away_team_name'] + ' (H)'
            first_df['home_team_name'] = first_df['home_team_name'] + ' (A)'
            first_df['away_team_name'].replace(team + ' (H)', np.nan,
                                               inplace=True)
            first_df['home_team_name'].replace(team + ' (A)', np.nan,
                                               inplace=True)
            first_df['next5'] = first_df['home_team_name'].fillna(
                first_df['away_team_name'])
            first_df.loc[first_df['blank'] == True, 'next5'] = 'BLANK'
            first_df['fdr'] = first_df['next5'].str[:3].map(teams_df.set_index(
                'short_name')['strength'])
            dup_df = first_df.duplicated(subset=['event2'],
                                         keep=False).reset_index()
            dup_df.columns = ['index', 'multiple']
            first_df = first_df.reset_index().merge(dup_df, on='index',
                                                    how='left')
            first_df = first_df[~((first_df['multiple'] == True) &
                                  (first_df['blank'] == True))]
            first_df['next5_new'] = first_df.groupby(['event2'])['next5'] \
                .transform(lambda x : ' + '.join(x))
            fdr_gw_aves = first_df[['event2', 'fdr']].groupby(
                'event2').mean().reset_index()
            fdr_gw_aves.columns = ['event2', 'fdr_gw_ave']
            first_df.drop_duplicates('event2', keep='first', inplace=True)
            first_df = first_df.merge(fdr_gw_aves, on='event2', how='left')
            sorted_df = pd.DataFrame(
                data={'event': np.arange(upc_gw, upc_gw+fixt_num)})
            sorted_df = sorted_df.merge(first_df, on='event', how='left')
            sorted_df['next5_new'].fillna('BLANK', inplace=True)
            new_df = get_new_df_from_sorted(upc_gw, sorted_df, team, fixt_num)
        else:
            rest_df = fixt_df[(fixt_df['home_team_name'] == team) |
                              (fixt_df['away_team_name'] == team)]
            rest_df.sort_values(['event2', 'kickoff_time'],
                                ascending=True, inplace=True)
            rest_df['away_team_name'] = rest_df['away_team_name'] + ' (H)'
            rest_df['home_team_name'] = rest_df['home_team_name'] + ' (A)'
            rest_df['away_team_name'].replace(team + ' (H)', np.nan,
                                              inplace=True)
            rest_df['home_team_name'].replace(team + ' (A)', np.nan,
                                              inplace=True)
            rest_df['next5'] = rest_df['home_team_name'].fillna(
                rest_df['away_team_name'])
            rest_df.loc[rest_df['blank'] == True, 'next5'] = 'BLANK'
            rest_df['fdr'] = rest_df['next5'].str[:3].map(teams_df.set_index(
                'short_name')['strength'])
            dup_df = rest_df.duplicated(subset=['event2'],
                                        keep=False).reset_index()
            dup_df.columns = ['index', 'multiple']
            rest_df = rest_df.reset_index().merge(dup_df, on='index',
                                                  how='left')
            rest_df = rest_df[~((rest_df['multiple'] == True) &
                                (rest_df['blank'] == True))]
            rest_df['next5_new'] = rest_df.groupby(['event2'])['next5'] \
                .transform(lambda x : ' + '.join(x))
            rest_fdr_gw_aves = rest_df[['event2', 'fdr']].groupby(
                'event2').mean().reset_index()
            rest_fdr_gw_aves.columns = ['event2', 'fdr_gw_ave']
            rest_df.drop_duplicates('event2', keep='first', inplace=True)
            rest_df = rest_df.merge(rest_fdr_gw_aves, on='event2', how='left')
            sorted_df2 = pd.DataFrame(
                data={'event': np.arange(upc_gw, upc_gw+fixt_num)})
            sorted_df2 = sorted_df2.merge(rest_df, on='event', how='left')
            sorted_df2['next5_new'].fillna('BLANK', inplace=True)
            two_df = get_new_df_from_sorted(upc_gw, sorted_df2, team, fixt_num)
            new_df = new_df.append(two_df, ignore_index=True)
            fdr_list = ['GW' + str(num) + '_fdr' for num in np.arange(
                upc_gw, upc_gw+fixt_num)]
            new_df['next5_fdr_ave'] = new_df[fdr_list].mean(axis=1)
            new_df.sort_values('next5_fdr_ave', ascending=True, inplace=True)
    plot_fdr_heatmap(new_df, upc_gw, fixt_num)
    return new_df


def get_top20_ppmillion(df, current_gw):
    full_df_current_gw = df[df['round'] == current_gw]
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
    elements_df = split_bootstrap_data('elements', ele_cols)
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

