import pandas as pd
import numpy as np
seed_data = pd.read_csv('data/historical_seeds.csv')
games = pd.read_csv('data/historical_games.csv')


seed_data['TeamIndex'] = seed_data.apply(lambda x: str(x['Season']) + '_' + str(x['TeamID']), axis=1)
seed_data['RawSeed'] = seed_data.apply(lambda x: x['Seed'][-2:], axis=1)
team_idx_to_seed = dict(zip(seed_data['TeamIndex'], seed_data['RawSeed']))



games['margin'] = games['WScore'] - games['LScore']
games['LIndex'] = games.apply(lambda row: str(row['Season']) + '_' + str(row['LTeamID']), axis=1)

for year in range(1998, 2019 + 1):
    df = games[games['Season'] == year]
    # find LIndex corresponding to maximum margin
    max_margin = df['margin'].max()
    loser_idx = df[df['margin'] == max_margin].LIndex.values[0]
    # find LIndex corresponding to minimum margin
    print(year, team_idx_to_seed[loser_idx])





