import numpy as np
from scipy.stats import poisson


# creates a (depth x depth) matrix of row:column full time result probabilities
# sums up all probabilities for home, draw, away
# returns odds given expected goals scored by both teams.
def return_odds(lambda_home, lambda_away, depth=11):
    goals_home = np.array([poisson.pmf(i, lambda_home) for i in range(depth)])
    goals_away = np.array([poisson.pmf(i, lambda_away) for i in range(depth)])
    result_matrix = np.outer(goals_home, goals_away)
    draw_prob = np.sum(np.diag(result_matrix))
    away_prob = np.sum(np.triu(result_matrix)) - draw_prob
    home_prob = np.sum(np.tril(result_matrix)) - draw_prob
    return home_prob, draw_prob, away_prob


# returns expected goals for each team given average goals scored/lost by both teams
# GSH - goals scored home
# GLH - goals lost home
# GSA - goals scored away
# GLA - goals lost away
# GP - games played
# see attached document for derivation of the below
def return_means(GSH, GLH, GSA, GLA, GP):
    return np.exp(np.log(GSH/GP) + np.log(GLA/GP)), np.exp(np.log(GSA/GP) + np.log(GLH/GP))


# returns bookmaker's margin given odds
def calculate_margin(odds_home, odds_draw, odds_away):
    return 1 / (1/odds_home + 1/odds_draw + 1/odds_away)


# returns probabilities implied by bookmaker's odds
def return_implied_prob(odds_home, odds_draw, odds_away):
    margin = calculate_margin(odds_home, odds_draw, odds_away)
    return margin/odds_home, margin/odds_draw, margin/odds_away


# returns probability of an outcome assuming the means are equal --
# to the goals scored by both teams
def find_maximum_probability(goals_home, goals_away):
    probs = return_odds(goals_home, goals_away)
    if goals_home > goals_away:
        return probs[0]
    elif goals_home == goals_away:
        return probs[1]
    elif goals_home < goals_away:
        return probs[2]