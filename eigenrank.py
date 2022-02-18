import numpy as np
import pandas as pd
from sportsreference.ncaab.teams import Teams
from sportsreference.ncaab.schedule import Schedule
from sportsreference.ncaab.boxscore import Boxscore
from datetime import datetime
import csv


game_features = ['index', 'team_abbr', 'opp_abbr', 'pf', 'pa', 'pace', 'location', 'datetime', 'true_pace']
team_idxs = {team_name: idx for idx, team_name in enumerate([team.name for team in Teams()])}

team_abbrevs = []
team_names = []
for team in Teams():
    team_abbrevs.append(team.abbreviation)
    team_names.append(team.name)

def has_game_happened(game):
    if game.points_for is None or game.points_against is None or game.datetime.date() >= datetime.today().date():
        return False
    else:
        return True

def has_pace(game):
    return game.boxscore.pace is not None

def calculate_pace(bs_index):
    # using old kenpom formula https://kenpom.com/blog/the-possession/
    # (FGA – OFFR) + TO + (Y * FTA)
    # Y = 0.44
    bs = Boxscore(bs_index)
    fga = bs.home_field_goal_attempts + bs.away_field_goal_attempts
    offr = bs.home_offensive_rebounds + bs.away_offensive_rebounds
    to = bs.home_turnovers + bs.away_turnovers
    y = 0.44
    fta = bs.home_free_throw_attempts + bs.away_free_throw_attempts
    est_pace = (fga - offr) + to + (y * fta)
    return est_pace


def load_data_from_csv(filename):
    # takes csv
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            data.append(row)
    return data

def update_data():
    # feature labels
    res = load_data_from_csv('data/game_data.csv') # could change to load
    tracked_games = [game[0] for game in res] # boxscore indices of all games already tracked

    for team in team_abbrevs:
        print(team)
        schedule = Schedule(team)
        for game in schedule:
            if game.boxscore_index in tracked_games:
                continue
            if has_game_happened(game):
                if has_pace(game):
                    pace = game.boxscore.pace
                    true_pace = True
                else:
                    pace = calculate_pace(game.boxscore_index)
                    true_pace = False
                data = [game.boxscore_index, team, game.opponent_abbr, game.points_for, game.points_against, pace, game.location, game.datetime.date(), true_pace]
                if data[0] not in tracked_games:
                    print(data)
                    res.append(data)
    return res

def write_data(filename, data_lst):
    # let first name data_lst be header
    header = game_features
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        assert len(header) == len(data_lst[0])
        writer.writerows(data_lst)

def data_to_df(array):
    return pd.DataFrame(array, columns=game_features)

def get_adj_mat(df):
    mat = np.array( len(df['team_abbr'].unique()), len(df['team_abbr'].unique()) )
    for idx, row in df.iterrows():
        margin = row['pf'] - row['pa']
        if row['location'].lower() == 'home':
            margin += hca_val #TODO
        elif row['location'].lower() == 'away':
            margin -= hca_val
        else: #neutral
            pass
        eff_margin = margin / row['pace']
        eff_margin = margin_utility(eff_margin)

        # each cell contains a row descirbiing all games of that matchup
        if len(mat[team_idxs[row['team_abbr']], team_idxs[row['opp_abbr']]]) == 0:
            mat[team_idxs[row['team_abbr']], team_idxs[row['opp_abbr']]] = [eff_margin]
            mat[team_idxs[row['opp_abbr']], team_idxs[row['team_abbr']]] = [1- eff_margin]
        else:
           mat[team_idxs[row['team_abbr']], team_idxs[row['opp_abbr']]].append(eff_margin)
           mat[team_idxs[row['opp_abbr']], team_idxs[row['team_abbr']]].append(1- eff_margin)
            
    return mat

def normalize_mat(mat):
    # itterate through array mat
    for i in range(len(mat)):
        num_games = sum([len(tup) for tup in mat[i]])
        for j in range(len(mat)):
            mat[i][j] = np.mean(mat[i][j]) / num_games
    return mat

def eigenrank(mat):
    from scipy.sparse.linalg import eigs
    val, vec = eigs(mat, which='LM', k=1)
    vec = np.ndarray.flatten(abs(vec))
    return [ [team_names[i], vec[i]] for i, team_name in range(len(team_names)) ]

def margin_utility(eff_margin):
    return 1 / (1 + np.exp(-eff_margin))

def main():
    game_data = update_data()
    write_data('data/game_data.csv', game_data)
    mat = normalize_mat(get_adj_mat(data_to_df(game_data)))
    r = eigenrank(mat)
    for key, value in r.items:
        print(key, value)

if __name__ == '__main__':
    main()
