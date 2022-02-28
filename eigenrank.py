import numpy as np
import pandas as pd
from sportsreference.ncaab.teams import Teams
from sportsreference.ncaab.schedule import Schedule
from sportsreference.ncaab.boxscore import Boxscore
from datetime import datetime
from scipy.sparse.linalg import eigs
import csv
from matplotlib import pyplot as plt

hca_val = 4 #TODO
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

def split_data(df, split_ratio=.8):
    # sort df by date
    df['datetime'] = pd.to_datetime(df['datetime'])
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
    # use rmse as error metric for now
    rmse = np.sqrt(np.mean((test_data['pred_margin'] - test_data['margin'])**2))
    print('rmse: ', rmse)
    return rmse, reg, df

def predict_margin(df, em_ratings_dict, recent_game_scores):
    from sklearn.linear_model import LinearRegression
    df['margin'] = df.apply(lambda row: row['pf'] - row['pa'], axis=1)
    df['team_rating'] = df.apply(lambda row: em_ratings_dict[row['team_abbr']], axis=1)
    df['opp_rating'] = df.apply(lambda row: em_ratings_dict[row['opp_abbr']], axis=1)
    # technically should be doing this after split TODO
    df['team_rating*opp_rating'] = df['team_rating'] * df['opp_rating']
    location_to_numeric = {'Home': 1, 'Away': -1, 'Neutral': 0}
    df['location_numeric'] = df.apply(lambda row: location_to_numeric[row['location']], axis=1)
    df['team_most_recent_1'] = df.apply(lambda row: recent_game_scores[row['team_abbr']][-1], axis=1)
    df['team_most_recent_2'] = df.apply(lambda row: recent_game_scores[row['team_abbr']][-2], axis=1)
    df['team_most_recent_3'] = df.apply(lambda row: recent_game_scores[row['team_abbr']][-3], axis=1)
    df['opp_most_recent_1'] = df.apply(lambda row: recent_game_scores[row['opp_abbr']][-1], axis=1)
    df['opp_most_recent_2'] = df.apply(lambda row: recent_game_scores[row['opp_abbr']][-2], axis=1)
    df['opp_most_recent_3'] = df.apply(lambda row: recent_game_scores[row['opp_abbr']][-3], axis=1)  

    train_data, test_data = split_data(df)
    
    # linear regression
    # location/hca should be dependent upon team
    train_X = train_data[['team_rating', 'opp_rating', 'team_rating*opp_rating', 'location_numeric', 'team_most_recent_1', 'team_most_recent_2', 'team_most_recent_3', 'opp_most_recent_1', 'opp_most_recent_2', 'opp_most_recent_3', 'pace']].values
    y = train_data['margin'].values/train_data['pace'].values
    reg = LinearRegression(fit_intercept=False).fit(train_X, y)

    # test
    print(test_data.head())
    test_data['pred_pace'] = test_data.apply(lambda row: predict_pace(df, row['team_abbr'], row['opp_abbr']), axis=1)
    test_X = test_data[['team_rating', 'opp_rating', 'team_rating*opp_rating', 'location_numeric', 'team_most_recent_1', 'team_most_recent_2', 'team_most_recent_3', 'opp_most_recent_1', 'opp_most_recent_2', 'opp_most_recent_3', 'pred_pace' ]].values
    # print(test_X)
    test_data['pred_net_efficiency'] = reg.predict(test_X)
    test_data['pred_margin'] = test_data['pred_net_efficiency'] * test_data['pred_pace']
    # use rmse as error metric for now
    rmse = np.sqrt(np.mean((test_data['pred_margin'] - test_data['margin'])**2))
    print('rmse: ', rmse)
    return rmse, reg, df

def show_rankings(r_lst):
    r_lst.sort(key=lambda x: x[1], reverse=True)
    for i, row in enumerate(r_lst):
        print(i + 1, row[0], round(row[1], 2))

def test_log_coef(train_df, test_df, plot=False, kvals = np.arange(0, 20, .5), choose=True):
    x = []
    y = []
    for k in kvals:
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

    # choose best x correspodning to lowest y
    if choose:
        best_k = x[y.index(max(y))]
        return best_k

def get_hca():
    pass

def game_score(bs_index, df, ratings_dict, team_name):
    game = df[df['index'] == bs_index]
    if game['team_abbr'].iloc[0] == team_name:
        margin = game['pf'].iloc[0] - game['pa'].iloc[0]
        if game['location'].iloc[0] == 'Home':
            margin -= hca_val
        elif game['location'].iloc[0] == 'Away':
            margin += hca_val
        margin = margin / game['pace'].iloc[0]
        margin_val = margin_utility(margin)
        opp_rating = ratings_dict[game['opp_abbr'].iloc[0]]
    else:
        margin = game['pa'].iloc[0] - game['pf'].iloc[0]
        if game['location'].iloc[0] == 'Home':
            margin += hca_val
        elif game['location'].iloc[0] == 'Away':
            margin -= hca_val
        margin = margin / game['pace'].iloc[0]
        margin_val = margin_utility(margin)
        opp_rating = ratings_dict[game['team_abbr'].iloc[0]]
    return margin_val * opp_rating

def transform_ratings(ratings_lst):
    return [[el[0], el[1]**.25 * 100] for el in ratings_lst]

def add_to_df(df, ratings_dict, em_ratings_dict):
    team_game_score = []
    opp_game_score = []
    team_eigen_rating = []
    opp_eigen_rating = []
    team_em_rating = []
    opp_em_rating = []
    for idx, row in df.iterrows():
        team_game_score.append(game_score(row['index'], df, ratings_dict, row['team_abbr']))
        opp_game_score.append(game_score(row['index'], df, ratings_dict, row['opp_abbr']))
        team_eigen_rating.append(ratings_dict[row['team_abbr']])
        opp_eigen_rating.append(ratings_dict[row['opp_abbr']])
        team_em_rating.append(em_ratings_dict[row['team_abbr']])
        opp_em_rating.append(em_ratings_dict[row['opp_abbr']])
    df['team_game_score'] = team_game_score
    df['opp_game_score'] = opp_game_score
    df['team_eigen_rating'] = team_eigen_rating
    df['opp_eigen_rating'] = opp_eigen_rating
    df['team_em_rating'] = team_em_rating
    df['opp_em_rating'] = opp_em_rating
    df['datetime'].astype('datetime64[ns]')
    return df

def get_game_scores_by_team(big_df):
    # sorted by latest to most recent
    res = {abbr:[] for abbr in big_df['team_abbr'].unique()}
    df = big_df
    for team in res.keys():
        team_df = df[(df['team_abbr'] == team) | (df['opp_abbr'] == team)]
        team_df = team_df.sort_values(by='datetime', ascending=True)
        for idx, row in team_df.iterrows():
            if row['team_abbr'] == team:
                res[team].append(row['team_game_score'])
            else:
                res[team].append(row['opp_game_score'])
    return res

def adj_em(big_df, reg, em_ratings_dict, recent_game_scores):
    cap = 10 #points
    res = {}
    teams = big_df['team_abbr'].unique()
    print(big_df.head())
    for team in teams:
        # TODO: include HCA
        errors = []
        team_df = big_df[(big_df['team_abbr'] == team) | (big_df['opp_abbr'] == team)]
        for idx, game in team_df.iterrows():
            pred_margin_per_poss = em_ratings_dict[game['team_abbr']] - em_ratings_dict[game['opp_abbr']]  / 100
            margin_per_poss = game['margin'] / game['pace']
            if game['team_abbr'] == team:
                pass
            else:
                pred_margin_per_poss = -pred_margin_per_poss
                margin_per_poss = -margin_per_poss
            error = pred_margin_per_poss - margin_per_poss
            error = error * 100
            if error < 0 and error < -cap:
                error = -cap
            elif error > 0 and error > cap:
                error = cap
            errors.append(error)
        res[team] = em_ratings_dict[team] + np.mean(errors)
        print(res[team], 'EM Rating ', em_ratings_dict[team], 'Adjustment: ', np.mean(errors))
    return res

def em_ratings_2(big_df, reg, em_ratings_dict, recent_game_scores):
    # these are really like recent em ratings because they factor in most recent games
    res = {}
    median_rating = np.median([v for v in em_ratings_dict.values()])
    median_most_recent_game_score = np.median([v[-1] for v in recent_game_scores.values()])
    for team in em_ratings_dict.keys():
        team_df = big_df[(big_df['team_abbr'] == team) | (big_df['opp_abbr'] == team)]
        team_mean_pace = np.mean(team_df['pace'])
        team_rating = em_ratings_dict[team]
        opp_rating = median_rating
        # technically should be doing this after split TODO
        team_rating_times_opp_rating = team_rating * opp_rating
        # location_to_numeric = {'Home': 1, 'Away': -1, 'Neutral': 0}
        location_numeric = 0
        team_most_recent_1 = recent_game_scores[team][-1]
        team_most_recent_2 = recent_game_scores[team][-2]
        team_most_recent_3 = recent_game_scores[team][-3]
        opp_most_recent_1 = median_most_recent_game_score
        opp_most_recent_2 = median_most_recent_game_score
        opp_most_recent_3 = median_most_recent_game_score
        pred_pace = team_mean_pace
        data = np.array([team_rating, opp_rating, team_rating_times_opp_rating, location_numeric, team_most_recent_1, team_most_recent_2, team_most_recent_3, opp_most_recent_1, opp_most_recent_2, opp_most_recent_3, pred_pace])
        df = pd.DataFrame(data=data.reshape(1, 11))
        margin_per_poss = reg.predict(df.values)
        res[team] = margin_per_poss[0]*100
    return res

def main():
    game_data = update_data(update_games=False)
    write_data('data/game_data.csv', game_data)
    df = data_to_df(game_data)
    train_df, test_df = split_data(df)
    best_coef = test_log_coef(train_df, test_df)

    mat = normalize_mat(get_adj_mat(df, k=best_coef))
    r_lst, r_dct = eigenrank(mat)
    r_lst = transform_ratings(r_lst)
    r_dct = {el[0]: el[1] for el in r_lst}
    rmse, reg, df = calc_margin_error(df, r_dct)

    # need to make its own function to get em_rating
    mean_rating = np.mean([row[1] for row in r_lst])
    neutral_location_numeric = 0
    em_ratings_dict = {}
    for row in r_lst:
        X = np.array([[row[1], mean_rating, row[1]*mean_rating, neutral_location_numeric]]).reshape(1, -1)
        em_rating = reg.predict(X)[0] * 100
        em_ratings_dict[row[0]] = em_rating
    em_rating_lst = [[k, v] for k, v in em_ratings_dict.items()]
    show_rankings(em_rating_lst)

    # seems like rating^(.25) is a good transformation for normally distributed ratings
    # plt.hist([el[1] for el in r_lst], bins=20)
    # plt.show()
    # show_rankings(r_lst)
    # print(win_error(df, ratings_dict=r_dct))

    # predicting
    big_df = add_to_df(df.copy(), r_dct, em_ratings_dict)
    game_scores_by_team_dict = get_game_scores_by_team(big_df)

    # adj_em_ratings_dict = adj_em(df, reg, em_ratings_dict, game_scores_by_team_dict)

    margin_pred_rmse, margin_pred_reg, margin_pred_df = predict_margin(big_df, em_ratings_dict, game_scores_by_team_dict)
    print(margin_pred_rmse)

    em_2 = em_ratings_2(big_df, margin_pred_reg, em_ratings_dict, game_scores_by_team_dict)
    show_rankings([[k, v] for k, v in em_2.items()])



if __name__ == '__main__':
    main()
