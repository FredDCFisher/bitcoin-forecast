import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np
from features import add_features
from data_fetch import fetch_bitcoin_data

def train_model(n_trees=100, max_depth=5, learning_rate=0.1, rf_weight=0.5):
    # Load and prepare data
    df = fetch_bitcoin_data()
    df = add_features(df)

    # What we want to predict — next day's closing price
    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    # Features the model learns from
    feature_cols = ['close', 'ma7', 'ma30', 'lag_1', 'lag_3', 'lag_7',
                    'daily_return', 'volatility_7', 'ma_crossover',
                    'gold_change', 'sp500_change', 'ftse_change', 'eth_change']

    X = df[feature_cols]
    y = df['target']

    # Split: 80% training, 20% testing — no peeking at future data!
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training on {len(X_train)} days, testing on {len(X_test)} days...")

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    # Train XGBoost
    xgb = XGBRegressor(n_estimators=n_trees, max_depth=max_depth,
                       learning_rate=learning_rate, random_state=42)
    xgb.fit(X_train, y_train)

    # Blend predictions using weights
    gb_weight = 1 - rf_weight
    rf_preds  = rf.predict(X_test)
    xgb_preds = xgb.predict(X_test)
    final_preds = (rf_weight * rf_preds) + (gb_weight * xgb_preds)

    # Evaluate
    mae  = mean_absolute_error(y_test, final_preds)
    rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    r2   = r2_score(y_test, final_preds)

    # Naive baseline — just predict yesterday's price
    naive_preds = X_test['close'].values
    naive_mae   = mean_absolute_error(y_test, naive_preds)

    print(f"\n--- Results ---")
    print(f"MAE:  ${mae:,.0f}  (naive: ${naive_mae:,.0f})")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"R²:   {r2:.3f}")
    print(f"Our model beats naive by: ${naive_mae - mae:,.0f}")

    return rf, xgb, feature_cols

if __name__ == "__main__":
    rf, xgb, features = train_model()