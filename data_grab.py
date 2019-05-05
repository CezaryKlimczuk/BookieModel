import pandas as pd
import numpy as np
import stat_tools as st

# IMPORTANT:
# curr_season and prev_seasons are data frames of games in a given season including data on the following things:
# 'HomeTeam', 'AwayTeam' - names of teams playing a certain match
# 'FTHG', 'FTAG' - full time goals scored by home/away team
# and more...


# creates a dictionary of teams playing in a current season
# sums up goals scored and lost and adds them to the dictionary
# if a team played in a lower class, it is assigned (40, 60), which is an average performance for promoted teams
# returns a dictionary of teams - team:(scored, lost)
def create_teams_dict(curr_season, prev_season):
    teams = dict.fromkeys(curr_season['HomeTeam'].drop_duplicates().tolist())
    for team in teams.keys():
        if team in prev_season['HomeTeam'].tolist():
            scored = prev_season[prev_season['HomeTeam'] == team]['FTHG'].sum() + prev_season[prev_season['AwayTeam'] == team]['FTAG'].sum()
            lost = prev_season[prev_season['HomeTeam'] == team]['FTAG'].sum() + prev_season[prev_season['AwayTeam'] == team]['FTHG'].sum()
            teams[team] = (scored, lost)
        else:
            teams[team] = (40, 60)
    return teams


# iterates through the games assigning modelled probabilities, and those implied by bookmakers
# updates teams' scored:lost ratio by replacing certain weight with the new result
# returns data frame with two sets of probability columns (modelled & implied by bookmakers)
def assign_probabilities(curr_season, teams):
    rounds = 2 * (len(teams) - 1)
    for index, row in curr_season.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        goal_means = st.return_means(teams[home][0],  # calculates poisson means used by the model
                                teams[home][1],
                                teams[away][0],
                                teams[away][1],
                                rounds)
        odds = st.return_odds(goal_means[0], goal_means[1])  # calculates probabilities using the poisson model
        curr_season.at[index, 'modelH'] = odds[0]  # creates modelX columns with projected probabilities
        curr_season.at[index, 'modelD'] = odds[1]
        curr_season.at[index, 'modelA'] = odds[2]
        implied_probs = st.return_implied_prob(row['BbAvH'], row['BbAvD'], row['BbAvA'])
        curr_season.at[index, 'probH'] = implied_probs[0]  # creates probX columns with implied probabilities
        curr_season.at[index, 'probD'] = implied_probs[1]
        curr_season.at[index, 'probA'] = implied_probs[2]
        i = 1  # update weight
        teams[home] = ((rounds - i)/rounds * teams[home][0] + i * row['FTHG'], (rounds - i)/rounds * teams[home][1] + i * row['FTAG'])
        teams[away] = ((rounds - i)/rounds * teams[away][0] + i * row['FTAG'], (rounds - i)/rounds * teams[away][1] + i * row['FTHG'])
    return curr_season


# iterates through games picking up probabilities for games' outcomes
# returns an aggregate log-likelihood for both model and bookmakers probabilities
# limit indicates how many games we're looking at (all of them by default)
def log_likelihood_sum(curr_season, limit=9999):
    log_bookies = 0
    log_model = 0
    for index, row in curr_season.iterrows():
        if row['FTR'] == 'H':
            log_bookies += np.log(row['probH'])
            log_model += np.log(row['modelH'])
        elif row['FTR'] == 'D':
            log_bookies += np.log(row['probD'])
            log_model += np.log(row['modelD'])
        elif row['FTR'] == 'A':
            log_bookies += np.log(row['probA'])
            log_model += np.log(row['modelA'])
        if index > limit:
            break
    return log_bookies, log_model


# returns the log-likelihood if someone were to bet at random
def random_choice_log_likelihood(teams):
    rounds = 2 * (len(teams) - 1)
    games = 0.5 * rounds * (rounds / 2 + 1)
    random = games * np.log(1 / 3)
    return random


# returns the log-likelihood assuming poisson means for every game are equal to actual scores
# this metrics can serve as a proxy for a "perfect model"
def return_optimal_log_likelihood(curr_season, limit=9999):
    log_optimal = 0
    for index, row in curr_season.iterrows():
        log_optimal += np.log(st.find_maximum_probability(row['FTHG'], row['FTAG']))
        if index > limit:
            break
    return log_optimal


# loads games data frame from a particular season and division
# returns log-likelihoods for bookmakers/model/equal probabilities
def compare_model(year, division):
    curr_season = pd.read_excel(f'hist_data/euro_data_{year - 1}{year}.xls', sheet_name=division)
    prev_season = pd.read_excel(f'hist_data/euro_data_{year - 2}{year - 1}.xls', sheet_name=division)
    teams = create_teams_dict(curr_season, prev_season)
    curr_season = assign_probabilities(curr_season, teams)
    bookies, model = log_likelihood_sum(curr_season)
    random = random_choice_log_likelihood(teams)
    optimal = return_optimal_log_likelihood(curr_season)
    return bookies, model, random, optimal