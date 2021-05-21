#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:18:47 2021

@author: timyouell
"""

base_url = 'https://fantasy.premierleague.com/api/'


ele_cols = ['id', 'web_name', 'team', 'team_code', 'element_type',
            'chance_of_playing_next_round', 'chance_of_playing_this_round',
            'code', 'cost_change_event', 'cost_change_event_fall',
            'cost_change_start', 'cost_change_start_fall', 'dreamteam_count',
            'element_type', 'ep_next', 'ep_this', 'event_points', 'form',
            'in_dreamteam', 'news', 'news_added', 'points_per_game',
            'selected_by_percent', 'status', 'transfers_in',
            'transfers_in_event', 'transfers_out', 'transfers_out_event',
            'value_form', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'goals_conceded', 'own_goals', 'penalties_saved',
            'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 'bonus',
            'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'influence_rank', 'influence_rank_type', 'creativity_rank',
            'creativity_rank_type', 'threat_rank', 'threat_rank_type',
            'ict_index_rank', 'ict_index_rank_type', 
            'corners_and_indirect_freekicks_order', 'direct_freekicks_order',
            'penalties_order', 'now_cost', 'total_points', 'value_season']


ele_types_cols = ['id', 'plural_name_short', 'squad_select', 'squad_min_play',
                  'squad_max_play', 'element_count']


teams_cols = ['code', 'id', 'name', 'short_name', 'strength',
              'strength_overall_home', 'strength_overall_away',
              'strength_attack_home', 'strength_attack_away',
              'strength_defence_home', 'strength_defence_away']


new_ele_cols = ['id', 'web_name', 'team', 'team_code', 'position_type',
                'chance_of_playing_next_round', 'chance_of_playing_this_round',
                'code', 'cost_change_event', 'cost_change_event_fall',
                'cost_change_start', 'cost_change_start_fall',
                'dreamteam_count', 'element_type', 'ep_next', 'ep_this',
                'event_points', 'form', 'in_dreamteam', 'news', 'news_added',
                'points_per_game', 'selected_by_percent', 'status',
                'transfers_in', 'transfers_in_event',
                'transfers_out', 'transfers_out_event', 'value_form',
                'minutes', 'goals_scored', 'assists', 'clean_sheets',
                'goals_conceded', 'own_goals', 'penalties_saved',
                'penalties_missed', 'yellow_cards', 'red_cards', 'saves',
                'bonus', 'bps', 'influence', 'creativity', 'threat',
                'ict_index', 'influence_rank', 'influence_rank_type',
                'creativity_rank', 'creativity_rank_type', 'threat_rank',
                'threat_rank_type', 'ict_index_rank', 'ict_index_rank_type',
                'corners_and_indirect_freekicks_order',
                'direct_freekicks_order', 'penalties_order', 'now_cost',
                'total_points', 'value_season']

cols_list = ['minutes', 'goals_scored', 'assists', 'goals_conceded', 'saves',
             'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
             'yellow_cards', 'red_cards', 'team_goals', 'team_conceded',
             'total_points']

headers = ['first_name', 'second_name', 'goals_scored', 'assists',
           'total_points', 'minutes', 'goals_conceded', 'creativity',
           'influence', 'threat', 'bonus', 'bps', 'ict_index', 'clean_sheets',
           'red_cards', 'yellow_cards', 'selected_by_percent', 'now_cost',
           'element_type']

cut_cols = ['total_points', 'round', 'was_home', 'player_minutes_FPGW',
            'total_minutes_FPGW', 'player_goals_scored_FPGW',
            'total_goals_scored_FPGW', 'player_assists_FPGW',
            'total_assists_FPGW', 'player_goals_conceded_FPGW',
            'total_goals_conceded_FPGW', 'player_saves_FPGW',
            'total_saves_FPGW', 'player_bonus_FPGW', 'total_bonus_FPGW',
            'player_bps_FPGW', 'total_bps_FPGW', 'player_influence_FPGW',
            'total_influence_FPGW', 'player_creativity_FPGW',
            'total_creativity_FPGW', 'player_threat_FPGW', 'total_threat_FPGW',
            'player_ict_index_FPGW', 'total_ict_index_FPGW',
            'player_yellow_cards_FPGW', 'total_yellow_cards_FPGW',
            'player_red_cards_FPGW', 'total_red_cards_FPGW',
            'player_team_goals_FPGW', 'total_team_goals_FPGW',
            'player_team_conceded_FPGW', 'total_team_conceded_FPGW',
            'player_total_points_FPGW', 'total_total_points_FPGW', 'player',
            'team_full', 'opponent_full', 'position', 'value', 'element',
            'player_value_FPGW']

team_stat_cols = ['strength', 'strength_overall_home', 'strength_overall_away',
                  'strength_attack_home', 'strength_attack_away',
                  'strength_defence_home', 'strength_defence_away']
