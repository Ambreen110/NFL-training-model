import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

def load_and_test(input_file, classifier_file, regressors_file):
    try:
        test_data = pd.read_csv(input_file)
        print(f"Test data loaded successfully from {input_file}.")
    except FileNotFoundError:
        print(f"File {input_file} not found in the current directory.")
        return
    
    required_columns = [
        'spread_line', 'away_spread_odds', 'home_spread_odds',
        'total_line', 'under_odds', 'over_odds',
        'away_moneyline', 'home_moneyline',
        'away_qb_name', 'home_qb_name',
        'home_win', 'result', 'home_score', 'away_score'
    ]
    
    missing_cols = [col for col in required_columns if col not in test_data.columns]
    if missing_cols:
        print(f"Missing columns in the test data: {missing_cols}")
        return

    feature_columns = [
        'spread_line', 'away_spread_odds', 'home_spread_odds',
        'total_line', 'under_odds', 'over_odds',
        'away_moneyline', 'home_moneyline',
        'away_qb_name', 'home_qb_name'
    ]
    X_test = test_data[feature_columns]
    
    try:
        home_win_classifier = joblib.load(classifier_file)
        score_regressors = joblib.load(regressors_file)
        print(f"Models loaded successfully from {classifier_file} and {regressors_file}.")
    except FileNotFoundError as e:
        print(f"Model file not found: {e.filename}")
        return
    
    home_win_prob = home_win_classifier.predict_proba(X_test)[:, 1] 
    home_win_pred = (home_win_prob > 0.5).astype(int)
    y_pred = score_regressors.predict(X_test)
    
    home_score_pred = y_pred[:, 1].round()
    away_score_pred = y_pred[:, 2].round()
    
    result_pred = (home_score_pred - away_score_pred).round()

    result_check = (home_score_pred - away_score_pred).round() == result_pred
    if not result_check.all():
        print("Warning: Some rows do not satisfy the condition 'home_score - away_score = result'.")
        print("Invalid rows:")
        print(test_data[~result_check])
    
    test_data['predicted_home_win'] = home_win_pred
    test_data['predicted_result'] = result_pred
    test_data['predicted_home_score'] = home_score_pred
    test_data['predicted_away_score'] = away_score_pred
    
    test_data['home_win_probability'] = home_win_prob * 100
    test_data['home_win_probability'] = test_data['home_win_probability'].round().astype(int).astype(str) + '%'
    
    if 'home_win' in test_data.columns and 'result' in test_data.columns and 'home_score' in test_data.columns and 'away_score' in test_data.columns:
        evaluation_data = test_data.dropna(subset=['home_win', 'result'])
        X_eval = evaluation_data[feature_columns]
        y_true = evaluation_data[['home_win', 'result', 'home_score', 'away_score']]
        
        home_win_accuracy = accuracy_score(y_true['home_win'], test_data.loc[evaluation_data.index, 'predicted_home_win'])
        print(f"Home Win Prediction Accuracy: {home_win_accuracy:.2f}")
        print("Home Win Classification Report:")
        print(classification_report(y_true['home_win'], test_data.loc[evaluation_data.index, 'predicted_home_win']))
        
        mse_result = mean_squared_error(y_true['result'], test_data.loc[evaluation_data.index, 'predicted_result'])
        print(f"Result Prediction Mean Squared Error: {mse_result:.2f}")

        mse_home_score = mean_squared_error(y_true['home_score'], test_data.loc[evaluation_data.index, 'predicted_home_score'])
        print(f"Home Score Prediction Mean Squared Error: {mse_home_score:.2f}")

        mse_away_score = mean_squared_error(y_true['away_score'], test_data.loc[evaluation_data.index, 'predicted_away_score'])
        print(f"Away Score Prediction Mean Squared Error: {mse_away_score:.2f}")
        
    else:
        print("Actual values for home_win, result, home_score, and away_score not found in test data.")
    
    output_file = '2024_predictions.csv'
    test_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    load_and_test('2024_data.csv', 'home_win_classifier.pkl', 'score_regressors.pkl')
