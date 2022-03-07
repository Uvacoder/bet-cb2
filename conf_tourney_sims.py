import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import random

rankings_df = pd.read_csv('data/rankings.csv')

# would be better not to hardode the labels
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

def sim_conf_tourney(name, conf_seed_dct, rounds_matchups, model, sims=10000):
    num_rounds = len(rounds_matchups)
    winner_counts = {name: 0 for name in conf_seed_dct.values()}
    placement_counts = {name : {round: 0 for round in rounds_matchups.keys()} for name in conf_seed_dct.values()}
    for sim in range(sims):
        seed_dct = conf_seed_dct.copy()
        for round_num, round in rounds_matchups.items():
            for game in round:
                (seed1, seed2) = game
                team1 = seed_dct[int(seed1)]
                team2 = seed_dct[int(seed2)]
                pred = predict_from_teams(team1, team2, model)
                x = random.random()
                if x < pred: # team1 wins
                    seed_dct.pop(int(seed2))
                else: # team2 wins
                    seed_dct[int(seed1)] = team2
                    seed_dct.pop(int(seed2))
            for team in seed_dct.values():
                placement_counts[team][round_num] += 1

        winner = seed_dct[1]
        winner_counts[winner] += 1
    
    for team in placement_counts.keys():
        for round in placement_counts[team].keys():
            placement_counts[team][round] /= sims

    df = pd.DataFrame.from_dict(placement_counts, orient='index')
    if num_rounds == 5:
        rename_dict = {1: 'round2', 2: 'qtrs', 3: 'semis', 4: 'finals', 5: 'champ'}
    elif num_rounds == 4:
        rename_dict = {1: 'qtrs', 2: 'semis', 3: 'finals', 4: 'champ'}
    elif num_rounds == 3:
        rename_dict = {1: 'qtrs', 2: 'semis', 3: 'champ'}

    df.rename(columns=rename_dict, inplace=True)
    df.insert(0, 'seed', conf_seed_dct.keys())
    df.insert(1, 'team', conf_seed_dct.values())
    df.set_index('seed', inplace=True)
    df.to_csv('data/tourney_sims/' + name + '.csv')
    print()
    print(name)
    print(df)
    return df


def acc_tourney(model, sims=10000, name='acc'):
    conf_seed_dct =  {1: 'duke', 2: 'notre dame', 3: 'north carolina', 4: 'miami fl', 5:'wake forest', 6: 'virginia', 7: 'virginia tech', 8: 'florida state', 9: 'syracuse', 10: 'clemson', 11: 'louisville', 12: 'pittsburgh', 13: 'boston college', 14: 'georgia tech', 15: 'north carolina state'}
    num_rounds = 5
    rounds_matchups = {i: [] for i in range(1, num_rounds + 1)}
    rounds_matchups[1] = [('12', '13'), ('10', '15'), ('11', '14')]
    rounds_matchups[2] = [('8', '9'), ('5', '12'), ('7', '10'), ('6', '11')]
    rounds_matchups[3] = [('1', '8'), ('2', '7'), ('3', '6'), ('4', '5')]
    rounds_matchups[4] = [('1', '4'), ('2', '3')]
    rounds_matchups[5] = [('1', '2')]
    return sim_conf_tourney(name, conf_seed_dct, rounds_matchups, model, sims)

def pac12_tourney(model, sims=10000, name='pac12'):
    conf_seed_dct = {1: 'arizona', 2: 'ucla', 3: 'southern california', 4: 'colorado', 5: 'oregon', 6: 'washington', 7: 'washington state', 8: 'arizona state', 9: 'stanford', 10: 'california', 11: 'utah', 12: 'oregon state'}
    num_rounds = 4
    rounds_matchups = {i: [] for i in range(1, num_rounds + 1)}
    rounds_matchups[1] = [('8', '9'), ('5', '12'), ('7', '10'), ('6', '11')]
    rounds_matchups[2] = [('1', '8'), ('2', '7'), ('3', '6'), ('4', '5')]
    rounds_matchups[3] = [('1', '4'), ('2', '3')]
    rounds_matchups[4] = [('1', '2')]
    return sim_conf_tourney(name, conf_seed_dct, rounds_matchups, model, sims)

def big_east_tourney(model, sims=10000, name='big_east'):
    conf_seed_dct = {1: 'providence', 2: 'villanova', 3: 'connecticut', 4: 'creighton', 5: 'marquette', 6: 'seton hall', 7: 'st johns ny', 8: 'xavier', 9: 'butler', 10: 'depaul', 11: 'georgetown'}
    num_rounds = 4
    rounds_matchups = {i: [] for i in range(1, num_rounds + 1)}
    rounds_matchups[1] = [('8', '9'), ('7', '10'), ('6', '11')]
    rounds_matchups[2] = [('1', '8'), ('2', '7'), ('3', '6'), ('4', '5')]
    rounds_matchups[3] = [('1', '4'), ('2', '3')]
    rounds_matchups[4] = [('1', '2')]
    return sim_conf_tourney(name, conf_seed_dct, rounds_matchups, model, sims)

def sec_tourney(model, sims=10000, name='sec'):
    conf_seed_dct = {1: 'auburn', 2: 'tennessee', 3: 'kentucky', 4: 'arkansas', 5: 'louisiana state', 6: 'alabama', 7: 'south carolina', 8: 'texas am', 9: 'florida', 10: 'mississippi state', 11: 'vanderbilt', 12: 'missouri', 13: 'mississippi', 14: 'georgia'}
    num_rounds = 5
    rounds_matchups = {i: [] for i in range(1, num_rounds + 1)}
    rounds_matchups[1] = [('12', '13'), ('11', '14')]
    rounds_matchups[2] = [('8', '9'), ('7', '10'), ('6', '11'), ('5', '12')]
    rounds_matchups[3] = [('1', '8'), ('2', '7'), ('3', '6'), ('4', '5')]
    rounds_matchups[4] = [('1', '4'), ('2', '3')]
    rounds_matchups[5] = [('1', '2')]
    return sim_conf_tourney(name, conf_seed_dct, rounds_matchups, model, sims)

def a10_tourney(model, sims=10000, name='a10'):
    conf_seed_dct = {1: 'davidson', 2: 'dayton', 3: 'st bonaventure', 4: 'virginia commonwealth', 5: 'saint louis', 6: 'richmond', 7: 'george washington', 8: 'fordham', 9: 'george mason', 10: 'massachusetts', 11: 'rhode island', 12: 'la salle', 13: 'saint josephs', 14: 'duquesne'}
    num_rounds = 5
    rounds_matchups = {i: [] for i in range(1, num_rounds + 1)}
    rounds_matchups[1] = [('12', '13'), ('11', '14')]
    rounds_matchups[2] = [('8', '9'), ('7', '10'), ('6', '11'), ('5', '12')]
    rounds_matchups[3] = [('1', '8'), ('2', '7'), ('3', '6'), ('4', '5')]
    rounds_matchups[4] = [('1', '4'), ('2', '3')]
    rounds_matchups[5] = [('1', '2')]
    return sim_conf_tourney(name, conf_seed_dct, rounds_matchups, model, sims)


def big12_tourney(model, sims=10000, name='big12'):
    conf_seed_dct = {1: 'kansas', 2: 'baylor', 3: 'texas tech', 4: 'texas', 5: 'texas christian', 6: 'iowa state', 7: 'oklahoma', 8: 'kansas state', 9: 'west virginia'}
    num_rounds = 4
    rounds_matchups = {i: [] for i in range(1, num_rounds + 1)}
    rounds_matchups[1] = [('8', '9')]
    rounds_matchups[2] = [('1', '8'), ('2', '7'), ('3', '6'), ('4', '5')]
    rounds_matchups[3] = [('1', '4'), ('2', '3')]
    rounds_matchups[4] = [('1', '2')]
    return sim_conf_tourney(name, conf_seed_dct, rounds_matchups, model, sims)

def big10_tourney(model, sims=10000, name='big10'):
    pass

def main():
    model = probability_win_model(games_df)
    acc_tourney(model)
    pac12_tourney(model)
    sec_tourney(model)
    a10_tourney(model)
    big_east_tourney(model)
    big12_tourney(model)
    big10_tourney(model)

if __name__ == '__main__':
    main()