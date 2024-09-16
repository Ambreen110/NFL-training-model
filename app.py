import pandas as pd
import nfl_data_py as nfl

def import_schedules(years):
   
    if not isinstance(years, (list, range)):
        raise ValueError('Input must be a list or range.')
    
    if min(years) < 1999:
        raise ValueError('Data not available before 1999.')
    
    scheds = pd.DataFrame()
    
    scheds = pd.read_csv('http://www.habitatring.com/games.csv')
    
    scheds = scheds[scheds['season'].isin(years)]
        
    return scheds

def extract_features(season):
    pbp = nfl.import_schedules([season])
    
    relevant_columns = [
        'away_team', 'home_team',
        'result',
        'spread_line', 'away_spread_odds', 'home_spread_odds',
        'total_line', 'under_odds', 'over_odds',
        'away_moneyline', 'home_moneyline',
        'surface', 'roof', 'location',
        'away_rest', 'home_rest',
        'away_qb_name', 'home_qb_name',
        'away_coach', 'home_coach',
        'stadium', 'home_score', 'away_score'
    ]
    
    features = pbp[relevant_columns].copy()
    
    required_columns = [
        'spread_line', 'away_spread_odds', 'home_spread_odds',
        'total_line', 'under_odds', 'over_odds',
        'away_moneyline', 'home_moneyline',
        'away_qb_name', 'home_qb_name', 'result', 'home_score'
    ]
    
    features['home_win'] = (features['result'] > 0).astype(int)
    
    features_filtered = features.dropna(subset=required_columns + ['home_win'])
    
    return features_filtered

def fetch_and_save_all_features(start_year, end_year):
    all_features = pd.DataFrame()
    
    seasons = list(range(start_year, end_year + 1))
    
    existing_file = 'all_data.csv'
    if pd.io.common.file_exists(existing_file):
        all_features = pd.read_csv(existing_file)
    
    for season in seasons:
        print(f"Processing season {season}...")
        features = extract_features(season)
        features['season'] = season 
        
        if not all_features.empty:
            features = features[~features.index.isin(all_features.index)]
        
        all_features = pd.concat([all_features, features], ignore_index=True)
    
    output_file = 'all_data.csv'
    all_features.to_csv(output_file, index=False)
    print(f"All features saved to {output_file}")

if __name__ == "__main__":
    fetch_and_save_all_features(2000, 2024)
