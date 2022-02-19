import numpy as np
import pandas as pd
from sportsreference.ncaab.teams import Teams
from sportsreference.ncaab.schedule import Schedule
from sportsreference.ncaab.boxscore import Boxscore
from datetime import datetime
from scipy.sparse.linalg import eigs
import csv
from matplotlib import pyplot as plt

hca_val = 3 #TODO
game_features = ['index', 'team_abbr', 'opp_abbr', 'pf', 'pa', 'pace', 'location', 'datetime', 'true_pace']
team_idxs = {team_name: idx for idx, team_name in enumerate([team.abbreviation for team in Teams()])}
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

def update_data(update_games=True):
    # feature labels
    res = load_data_from_csv('data/game_data.csv') # could change to load
    if not update_games:
        return res
    tracked_games = [game[0] for game in res] # boxscore indices of all games already tracked

    for team in team_abbrevs:
        schedule = Schedule(team)
        for game in schedule:
            if game.opponent_abbr.upper() not in team_abbrevs:
                continue
            if game.boxscore_index in tracked_games:
                continue
            if has_game_happened(game):
                if has_pace(game):
                    pace = game.boxscore.pace
                    true_pace = True
                else:
                    pace = calculate_pace(game.boxscore_index)
                    true_pace = False
                data = [game.boxscore_index, team, game.opponent_abbr.upper(), game.points_for, game.points_against, pace, game.location, game.datetime.date(), true_pace]
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
    return pd.DataFrame(array, columns=game_features).astype({'index': str, 'pf': int, 'pa': int, 'pace': float, 'true_pace': bool})

def get_adj_mat(df, k=4):
    mat = np.empty(shape=(len(team_abbrevs), len(team_abbrevs)), dtype=list)
    mat.fill([])
    for idx, row in df.iterrows():
        margin = row['pf'] - row['pa']
        if row['location'].lower() == 'home':
            margin += hca_val #TODO
        elif row['location'].lower() == 'away':
            margin -= hca_val
        else: #neutral
            pass
        eff_margin = margin / row['pace']
        eff_margin = margin_utility(eff_margin, k)

        # each cell contains a row descirbiing all games of that matchup
        if len(mat[team_idxs[row['team_abbr']], team_idxs[row['opp_abbr']]]) == 0:
            mat[team_idxs[row['team_abbr']], team_idxs[row['opp_abbr']]] = [eff_margin]
            mat[team_idxs[row['opp_abbr']], team_idxs[row['team_abbr']]] = [1 - eff_margin]
        else:
           mat[team_idxs[row['team_abbr']], team_idxs[row['opp_abbr']]].append(eff_margin)
           mat[team_idxs[row['opp_abbr']], team_idxs[row['team_abbr']]].append(1 - eff_margin)       
    return mat

def normalize_mat(mat):
    # iterate through array mat
    for i in range(len(mat)):
        num_games = sum([len(tup) for tup in mat[i]])
        for j in range(len(mat)):
            if len(mat[i][j]) == 0:
                mat[i][j] = 0
            else:
                mat[i][j] = np.sum(mat[i][j]) / num_games
    return mat.astype(float)

def eigenrank(mat):
    val, vec = eigs(mat, which='LM', k=1)
    vec = np.ndarray.flatten(abs(vec))
    res = {abbr: vec[i] for i, abbr in enumerate(team_abbrevs) }
    return  [[abbr, vec[i]] for i, abbr in enumerate(team_abbrevs) ], {abbr: vec[i] for i, abbr in enumerate(team_abbrevs) }

def margin_utility(eff_margin, k=4):
    return 1 / (1 + np.exp(-k * eff_margin))

def split_data(df, split_ratio=.7):
    # sort df by date
    df = df.sort_values(by='datetime', ascending=True)
    train_df = df.iloc[:int(len(df) * split_ratio)]
    test_df = df.iloc[int(len(df) * split_ratio):]
    return train_df, test_df

def get_pred_winner_no_hca(row, ratings_dict):
    return row['team_abbr'] if ratings_dict[row['team_abbr']] > ratings_dict[row['opp_abbr']] else row['opp_abbr']

def get_true_winner(row):
    return row['team_abbr'] if row['pf'] > row['pa'] else row['opp_abbr']

def predict_wins(df1, ratings_dict, hca=False):
    if not hca:
        df1['win_pred'] = df1.apply(lambda row: get_pred_winner_no_hca(row, ratings_dict), axis=1)
    else:
        pass
    df1['winner'] = df1.apply(lambda row: get_true_winner(row), axis=1)
    df1['correct'] = (df1['winner'] == df1['win_pred']).astype(int)
    return df1

def win_error(test_data, ratings_dict, hca=False):
    df_res = predict_wins(test_data, ratings_dict, hca)
    return df_res['correct'].mean()

def predict_pace(df, team1, team2):
    # select dataframe where either team_abbr is team1 or opp_abbr is team1
    team1_avg_pace = df[(df['team_abbr'] == team1) | (df['opp_abbr'] == team1)]['pace'].mean()
    team2_avg_pace = df[(df['team_abbr'] == team2) | (df['opp_abbr'] == team2)]['pace'].mean()
    # team1_avg_page = df[df['team_abbr'] == team1 or df['opp_abbr'] == team2]['pace'].mean()
    # team2_avg_page = df[team2 in df[['team_abbr', 'opp_abbr']]]['pace'].mean()
    return np.mean([team1_avg_pace, team2_avg_pace])

def calc_margin_error(df, ratings_dict):
    from sklearn.linear_model import LinearRegression
    df['margin'] = df.apply(lambda row: row['pf'] - row['pa'], axis=1)
    df['team_rating'] = df.apply(lambda row: ratings_dict[row['team_abbr']], axis=1)
    df['opp_rating'] = df.apply(lambda row: ratings_dict[row['opp_abbr']], axis=1)
    # technically should be doing this after split TODO
    df['team_rating*opp_rating'] = df['team_rating'] * df['opp_rating']
    location_to_numeric = {'Home': 1, 'Away': -1, 'Neutral': 0}
    df['location_numeric'] = df.apply(lambda row: location_to_numeric[row['location']], axis=1)

    train_data, test_data = split_data(df)
    
    # linear regression
    # location/hca should be dependent upon team
    train_X = train_data[['team_rating', 'opp_rating', 'team_rating*opp_rating', 'location_numeric']].values
    y = train_data['margin'].values/train_data['pace'].values
    reg = LinearRegression(fit_intercept=False).fit(train_X, y)

    # test
    print(test_data.head())
    test_data['pred_pace'] = test_data.apply(lambda row: predict_pace(df, row['team_abbr'], row['opp_abbr']), axis=1)
    test_X = test_data[['team_rating', 'opp_rating', 'team_rating*opp_rating', 'location_numeric']].values
    # print(test_X)
    test_data['pred_net_efficiency'] = reg.predict(test_X)
    test_data['pred_margin'] = test_data['pred_net_efficiency'] * test_data['pred_pace']
    print(reg.predict(np.array([40, 40, 40*40, 0]).reshape(1, -1)))
    # use rmse as error metric for now
    rmse = np.sqrt(np.mean((test_data['pred_margin'] - test_data['margin'])**2))
    print('rmse: ', rmse)
    return rmse, reg, df

def show_rankings(r_lst):
    r_lst.sort(key=lambda x: x[1], reverse=True)
    for i, row in enumerate(r_lst):
        print(i + 1, row[0], round(row[1], 2))

def test_log_coef(train_df, test_df, plot=True):
    x = []
    y = []
    for k in np.arange(0, 100, 1):
        mat = normalize_mat(get_adj_mat(train_df, k))
        r_lst, ratings_dict = eigenrank(mat)
        x.append(k)
        y.append(win_error(test_df, ratings_dict))
        print(x[-1], round(y[-1]* 100, 2))
    if plot:
        plt.plot(x, y)
        plt.xlabel('k')
        plt.ylabel('win rate')
        plt.show()
        plt.close()

def transform_ratings(ratings_lst):
    return [[el[0], el[1]**.25 * 100] for el in ratings_lst]

def main():
    game_data = update_data(update_games=True)
    write_data('data/game_data.csv', game_data)
    df = data_to_df(game_data)
    train_df, test_df = split_data(df)
    # test_log_coef(train_df, test_df)

    mat = normalize_mat(get_adj_mat(df))
    r_lst, r_dct = eigenrank(mat)
    r_lst = transform_ratings(r_lst)
    r_dct = {el[0]: el[1] for el in r_lst}
    rmse, reg, df = calc_margin_error(df, r_dct)
    print(df.head(10))
    mean_rating = np.mean([row[1] for row in r_lst])
    print(mean_rating)
    neutral_location_numeric = 0
    em_ratings = {}
    for row in r_lst:
        X = np.array([[row[1], mean_rating, row[1]*mean_rating, neutral_location_numeric]]).reshape(1, -1)
        em_rating = reg.predict(X)[0] * 100
        em_ratings[row[0]] = em_rating
    em_rating_lst = [[k, v] for k, v in em_ratings.items()]
    show_rankings(em_rating_lst)

    # seems like rating^(.25) is a good transformation for normally distributed ratings
    plt.hist([el[1] for el in r_lst], bins=20)
    plt.show()
    # show_rankings(r_lst)
    # print(win_error(df, ratings_dict=r_dct))

if __name__ == '__main__':
    main()
