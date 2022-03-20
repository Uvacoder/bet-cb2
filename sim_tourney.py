from re import L
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
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


def simulate_tourney(games_df, model, sims = 100000):
    num_rounds = 6
    teams = []
    with open('seeds.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            teams.append(row[3])
    placement_counts = {name : {r: 0 for r in range(1, num_rounds + 1)} for name in teams}

    for sim in range(sims):
        seeds = {'w': {}, 's': {}, 'e': {}, 'm': {}}
        with open('seeds.csv', 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'Team':
                    continue
                div = row[2]
                seed = int(row[1])
                name = row[3]
                if seed not in seeds[div].keys():
                    seeds[div][seed] = [name]
                else:
                    seeds[div][seed].append(name)

        rounds = {}
        rounds[1] = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
        rounds[2] = [(1, 8), (2, 7), (3, 6), (4, 5)]
        rounds[3] = [(1, 4), (2, 3)]
        rounds[4] = [(1, 2)]

        seeds_by_div = dict(seeds)
        divs = ['w', 's', 'e', 'm']
        for div in divs:
            seeds = seeds_by_div[div]
            # first four
            for key, value in seeds.items():
                if len(value) == 2:
                    [team1, team2] = value
                    pred = predict_from_teams(team1, team2, model)
                    x = random.random()
                    if x < pred:
                        seeds[key] = [team1]
                    else:
                        seeds[key] = [team2]

            seeds = {k: v[0] for k, v in seeds.items()}
            for round_num in range(1, 4 + 1):
                round = rounds[round_num]
                for game in round:
                    (seed1, seed2) = game
                    team1 = seeds[seed1]
                    team2 = seeds[seed2]
                    pred = predict_from_teams(team1, team2, model)
                    x = random.random()
                    if x < pred:
                        seeds.pop(seed2)
                    else:
                        seeds[seed1] = team2
                        seeds.pop(seed2)
                for i, team in seeds.items():
                    # print(i, team)
                    placement_counts[team][round_num] += 1
            seeds_by_div[div] = seeds[1]

        # reseed final four
        seeds = {}
        seeds[1] = seeds_by_div['w']
        seeds[4] = seeds_by_div['e']
        seeds[2] = seeds_by_div['s']
        seeds[3] = seeds_by_div['m']

        rounds[5] = [(1, 4), (2, 3)]
        rounds[6] = [(1, 2)]

        for round_num in range(5, 6+1):
            round = rounds[round_num]
            for game in round:
                (seed1, seed2) = game
                team1 = seeds[seed1]
                team2 = seeds[seed2]
                pred = predict_from_teams(team1, team2, model)
                x = random.random()
                if x < pred:
                    seeds.pop(seed2)
                else:
                    seeds[seed1] = team2
                    assert team2 in seeds.values()
                    seeds.pop(seed2)
            for team in seeds.values():
                placement_counts[team][round_num] += 1

    
    for team in placement_counts.keys():
        for round in placement_counts[team].keys():
            placement_counts[team][round] /= sims

    df = pd.DataFrame.from_dict(placement_counts, orient='index')
    rename_dict = {0: 'round64', 1: 'round32', 2: 'sweet16', 3: 'elite8', 4: 'final4', 5: 'championship', 6: 'champ'}

    df.rename(columns=rename_dict, inplace=True)
    # print(len(seeds.keys()))
    # print(len(seeds.values()))
    # print(df.shape)
    # print(len(placement_counts))
    df.to_csv('data/tourney_sim' + '.csv')
    print()
    print(df)
    return df

if __name__ == '__main__':
    simulate_tourney(games_df, probability_win_model(games_df))
    