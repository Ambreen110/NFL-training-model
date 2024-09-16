import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor # Added to handle multiple regression targets together
import joblib

def process_and_train_model(input_file):
    try:
        features_df = pd.read_csv(input_file)
        print(f"Data loaded successfully from {input_file}.")
    except FileNotFoundError:
        print(f"File {input_file} not found in the current directory.")
        return
    
    if not features_df.empty:
        print("Initial Data Overview:")
        print(features_df.head())
        print(features_df.info())

        bool_columns = features_df.select_dtypes(include='bool').columns
        features_df[bool_columns] = features_df[bool_columns].astype(int)

        feature_prefixes = ['away_team_', 'home_team_', 'location_', 'surface_', 'roof_', 'stadium_']
        categorical_features = [col for col in features_df.columns if any(col.startswith(prefix) for prefix in feature_prefixes)]

        feature_columns = ['spread_line', 'away_spread_odds', 'home_spread_odds', 'total_line', 'under_odds', 'over_odds', 'away_moneyline', 'home_moneyline'] + categorical_features
        target_columns = ['home_win', 'result', 'home_score', 'away_score']
        
        missing_cols = [col for col in feature_columns + target_columns if col not in features_df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return
        
        X = features_df[feature_columns]
        y = features_df[target_columns]

        if y.isnull().any().any():
            print("NaN values found in target variables. Removing rows with NaN values.")
            features_df = features_df.dropna(subset=target_columns)
            X = features_df[feature_columns]
            y = features_df[target_columns]

        numeric_features = ['spread_line', 'away_spread_odds', 'home_spread_odds', 'total_line', 'under_odds', 'over_odds', 'away_moneyline', 'home_moneyline']
        categorical_features = [col for col in X.columns if col not in numeric_features]

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        clf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
# Comment: A pipeline specifically for the 'home_win' classifier was maintained, 
# as classification models and regression models cannot be merged in one pipeline. 

        reg_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
        ])
# Comment: A single pipeline is created for the regression task, handling all regression targets ('result', 
# 'home_score', and 'away_score') with MultiOutputRegressor, allowing the model to handle multiple 
# regression targets simultaneously instead of separate pipelines as in the previous code.

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf_pipeline.fit(X_train, y_train['home_win'])
        home_win_pred = clf_pipeline.predict(X_test)
        
        reg_pipeline.fit(X_train, y_train[['result', 'home_score', 'away_score']])
        y_pred = reg_pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test['home_win'], home_win_pred)
        print(f"Home Win Prediction Accuracy: {accuracy:.2f}")
        print("Home Win Classification Report:")
        print(classification_report(y_test['home_win'], home_win_pred))

        mse_result = mean_squared_error(y_test['result'], y_pred[:, 0])
        print(f"Result Prediction Mean Squared Error: {mse_result:.2f}")

        mse_home_score = mean_squared_error(y_test['home_score'], y_pred[:, 1])
        print(f"Home Score Prediction Mean Squared Error: {mse_home_score:.2f}")

        mse_away_score = mean_squared_error(y_test['away_score'], y_pred[:, 2])
        print(f"Away Score Prediction Mean Squared Error: {mse_away_score:.2f}")

        joblib.dump(clf_pipeline, 'home_win_classifier.pkl') # Changed model name
        joblib.dump(reg_pipeline, 'score_regressors.pkl') # Changed model name
        print("Models saved as 'home_win_classifier.pkl' and 'score_regressors.pkl'")
    else:
        print("No data to process.")

if __name__ == "__main__":
    process_and_train_model('all_data.csv')
