import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeRegressor
import random
import csv

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
# games_df = games_df.drop(labels=['pf', 'pa', 'location', 'team_abbr', 'opp_abbr'], axis=1)


preseason_ap_25 = ['gonzaga', 'ucla', 'texas', 'michigan', 'vilanova', 'kansas', 'purude', 'baylor', 'duke', 'kentucky', 'illinois', 'memphis', 'oregon', 'alabama', 'houston', 'arkansas', 'ohio state', 'tennessee', 'north carolina', 'florida state', 'maryland', 'auburn', 'st bonaventure', 'connecticut', 'virginia']
games_df['preseason_ranked_team'] = games_df.apply(lambda x: 1 if x['team_abbr'] in preseason_ap_25 else 0, axis=1)
games_df['preseason_ranked_opp'] = games_df.apply(lambda x: 1 if x['opp_abbr'] in preseason_ap_25 else 0, axis=1)


def probability_win_model(games_df):
    """
    Calculate log odds of winning for each team in the rankings dataframe
    """
    
    features = ['team_em', 'opp_em', 'location_numeric', 'preseason_ranked_team', 'preseason_ranked_opp']
    label = ['team_wins']

    X = games_df[features].values
    y = np.array(games_df[label].values).ravel()

    # logreg = SGDClassifier(loss='log')
    logreg = LogisticRegression()
    logreg.fit(X, y)
    model = logreg

    # model = DecisionTreeRegressor(max_depth=5)
    # model.fit(X, y)

    return model

def predict_game(s_x, model):
    return model.predict_proba(s_x)[0][1]
    # return model.predict(s_x)[0]

def predict_games(games_df, model):
    features = ['team_em', 'opp_em', 'location_numeric']
    # label = ['winner']
    X = games_df[features].values
    preds = model.predict_proba(X)
    return [pred[1] for pred in preds]

def predict_from_teams(team, opp, model, loc_numeric=0):
    team_em = rankings_dict[team]
    opp_em = rankings_dict[opp]
    team_ps_ranked = 1 if team in preseason_ap_25 else 0
    opp_ps_ranked = 1 if opp in preseason_ap_25 else 0
    s_x = np.array([team_em, opp_em, loc_numeric, team_ps_ranked, opp_ps_ranked]).reshape(1, -1)
    return predict_game(s_x, model)

team_spelling_to_id = dict()
with open('data/MTeamSpellings.csv', 'r', encoding='cp1252') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
        team_spelling_to_id[row[0]] = row[1]
    team_spelling_to_id['saint marys ca'] = team_spelling_to_id['saint-marys']
    team_spelling_to_id['saint peters'] = team_spelling_to_id['saint-peters']
    team_spelling_to_id['texas am corpus christi'] = team_spelling_to_id['texas-am-corpus-christi']
    team_spelling_to_id['loyola il'] = team_spelling_to_id['loyola-il']
    team_spelling_to_id['alabama birmingham'] = team_spelling_to_id['alabama-birmingham']

teams = []
with open('seeds.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            teams.append(row[3])
        except:
            continue


model = probability_win_model(games_df)
results = {}
for team1 in teams:
    id1 = team_spelling_to_id[team1]
    for team2 in teams:
        if team1 == team2:
            continue
        id2 = team_spelling_to_id[team2]
        if id1 >= id2:
            continue
        pred = predict_from_teams(team1, team2, model)
        print(team1, team2, str(round(pred*100, 2)))
        id_str = str(2022) + '_' + str(id1) + '_' + str(id2)
        results[id_str] = pred


with open('kaggle_sub1.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Id', 'Pred'])
    for key, value in results.items():
        writer.writerow([key, value])

