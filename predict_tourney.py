import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
import random
from sportsreference.ncaab.teams import Teams
import csv
import datetime

rankings_df = pd.read_csv('data/rankings.csv')

# would be better not to hard code the labels
rankings_df = rankings_df.drop(labels=['adj em rating', 'conference', 'wins', 'losses'], axis=1)
rankings_df['em'] = rankings_df['em with recency bias']
rankings_df = rankings_df.drop(labels=['em with recency bias'], axis=1)
rankings_dict = dict(zip(rankings_df.team, rankings_df.em))

games_df = pd.read_csv('data/game_data.csv')
location_to_numeric = {'Home': 1, 'Away': -1, 'Neutral': 0}
games_df = games_df[['team_abbr', 'opp_abbr', 'pf', 'pa', 'location']]
games_df['team_abbr'] = games_df['team_abbr'].apply(lambda x: x.lower().replace('-', ' '))
games_df['opp_abbr'] = games_df['opp_abbr'].apply(lambda x: x.lower().replace('-', ' '))
games_df['team_wins'] = games_df.apply(lambda x: int(x['pf'] > x['pa']), axis=1)
games_df['location_numeric'] = games_df['location'].apply(lambda x: location_to_numeric[x])
games_df['team_em'] = games_df.apply(lambda x: rankings_dict[x['team_abbr']], axis=1)
games_df['opp_em'] = games_df.apply(lambda x: rankings_dict[x['opp_abbr']], axis=1)
games_df['margin'] = games_df.apply(lambda x: x['pf'] - x['pa'], axis=1)
games_df = games_df.drop(labels=['pf', 'pa', 'location', 'team_abbr', 'opp_abbr'], axis=1)


def probability_win_model(games_df):
    """
    Calculate log odds of winning for each team in the rankings dataframe
    """
    features = ['team_em', 'opp_em', 'location_numeric']
    label = ['team_wins']

    X = games_df[features].values
    y = np.array(games_df[label].values).ravel()

    # logreg = SGDClassifier(loss='log')
    logreg = LogisticRegression()
    logreg.fit(X, y)
    return logreg

def spread_model(games_df):
    """
    Calculate log odds of winning for each team in the rankings dataframe
    """
    features = ['team_em', 'opp_em', 'location_numeric']
    label = ['margin']

    X = games_df[features].values
    y = np.array(games_df[label].values).ravel()

    # logreg = SGDClassifier(loss='log')
    linreg = LinearRegression()
    linreg.fit(X, y)
    return linreg

def predict_game(s_x, model):
    return model.predict_proba(s_x)[0][1]

def predict_games(games_df, model):
    features = ['team_em', 'opp_em', 'location_numeric']
    # label = ['winner']
    X = games_df[features].values
    preds = model.predict_proba(X)
    return [pred[1] for pred in preds]

def predict_from_teams(team, opp, model, loc_numeric=0):
    team_em = rankings_dict[team]
    opp_em = rankings_dict[opp]
    s_x = np.array([team_em, opp_em, loc_numeric]).reshape(1, -1)
    return predict_game(s_x, model)

def predict_spread(s_x, model):
    return model.predict(s_x)[0]

def predict_spread_from_teams(team, opp, model, loc_numeric=0):
    team_em = rankings_dict[team]
    opp_em = rankings_dict[opp]
    s_x = np.array([team_em, opp_em, loc_numeric]).reshape(1, -1)
    return predict_spread(s_x, model)


tourney_teams = []
with open ('data/tourney_sim_google_sheet.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        tourney_teams.append(line[1])

seed_dct = {}
with open ('data/tourney_sim_google_sheet.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for line in reader:
        print(line)
        if line[0].startswith('s'):
            continue
        if len(line[0]) == 0:
            break
        seed_dct[line[1]] = int(line[0])


upcoming_games = []
for team in Teams():
    print(team.abbreviation.replace('-', ' ').lower())
    if team.abbreviation.replace('-', ' ').lower() in tourney_teams:
        for game in team.schedule:
            print(team.name, game.opponent_name)
            if game.datetime.date() >= datetime.datetime.now().date():# game hasn't happened yet
                upcoming_games.append([team.abbreviation.replace('-', ' ').lower(), game.opponent_abbr.replace('-', ' ').lower()])

print(tourney_teams)
win_model = probability_win_model(games_df)
spread_model = spread_model(games_df)
win_preds = []
for game in upcoming_games:
    win_preds.append([seed_dct[game[0]], game[0], seed_dct[game[1]], game[1], predict_from_teams(game[0], game[1], win_model), predict_spread_from_teams(game[0], game[1], spread_model)])

with open('data/tourney_win_preds.csv', 'w') as f:
    writer = csv.writer(f)
    for row in win_preds:
        writer.writerow(row)

# sort win_preds by seed
win_preds = sorted(win_preds, key=lambda x: x[0])

# turn array into html table
html_string = ''
for row in win_preds:
    if row[-1] < 0:
        continue
    html_string += '<p>'
    line_string = '({}) {} vs. ({}) {}<br>'.format(row[0], row[1], row[2], row[3])
    line_string += 'favorite: ({}) {}<br>'.format(row[0], row[1])
    line_string += 'win probability: {:.1f}%<br>'.format(row[4]*100)
    line_string += 'spread: {:.1f}'.format(row[5])
    html_string += line_string
    html_string += '</p>'


print(html_string)


