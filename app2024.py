import nfl_data_py as nfl
import pandas as pd

import pandas as pd

def extract_features(season):
    pbp = nfl.import_schedules([season])
    
    relevant_columns = [
        'away_team', 'home_team',
        'result', 'home_score', 'away_score',
        'spread_line', 'away_spread_odds', 'home_spread_odds',
        'total_line', 'under_odds', 'over_odds',
        'away_moneyline', 'home_moneyline',
        'surface', 'roof', 'location',
        'away_rest', 'home_rest',
        'away_qb_name', 'home_qb_name',
        'away_coach', 'home_coach',
        'stadium'
    ]
    
    features = pbp[relevant_columns].copy()
    
    print("Unique values in the 'result' column:", features['result'].unique())
    
    features['home_win'] = features['result'].apply(
        lambda x: 1 if pd.notna(x) and x > 0 else (0 if pd.notna(x) and x <= 0 else pd.NA)
    )
    
    required_columns = [
        'spread_line', 'away_spread_odds', 'home_spread_odds',
        'total_line', 'under_odds', 'over_odds',
        'away_moneyline', 'home_moneyline',
        'away_qb_name', 'home_qb_name'
    ]
    
    features_filtered = features.dropna(subset=required_columns)
    
    return features_filtered

def fetch_and_save_all_features(seasons):
    all_features = pd.DataFrame()
    
    for season in seasons:
        print(f"Processing season {season}...")
        features = extract_features(season)
        
        features['season'] = season 
        all_features = pd.concat([all_features, features], ignore_index=True)
    
    output_file = '2024_data.csv'
    all_features.to_csv(output_file, index=False)
    print(f"All features saved to {output_file}")
    
    return all_features

if __name__ == "__main__":
    seasons = [2024]
    fetch_and_save_all_features(seasons)
