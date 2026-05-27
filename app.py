import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from data_fetch import fetch_bitcoin_data
from features import add_features

st.set_page_config(page_title="Bitcoin Forecaster", page_icon="₿", layout="wide")
st.title("₿ Bitcoin Price Forecaster")
st.markdown("Adjust the model parameters in the sidebar and see how predictions change.")

# Sidebar controls
st.sidebar.header("Model Parameters")
n_trees      = st.sidebar.slider("Number of Trees",  50, 500, 100, step=50)
max_depth    = st.sidebar.slider("Max Depth",         3, 10,  5)
learning_rate = st.sidebar.slider("Learning Rate",   0.01, 0.3, 0.1, step=0.01)
rf_weight    = st.sidebar.slider("Random Forest Weight", 0.0, 1.0, 0.5, step=0.1)
gb_weight    = 1 - rf_weight
st.sidebar.markdown(f"XGBoost weight: **{gb_weight:.1f}**")

# Load data
@st.cache_data
def load_data():
    df = fetch_bitcoin_data()
    df = add_features(df)
    return df

with st.spinner("Loading data..."):
    df = load_data()

# Prepare features
feature_cols = ['close', 'ma7', 'ma30', 'lag_1', 'lag_3', 'lag_7',
                'daily_return', 'volatility_7', 'ma_crossover',
                'gold_change', 'sp500_change', 'ftse_change', 'eth_change']

df['target'] = df['close'].shift(-1)
df = df.dropna()

X = df[feature_cols]
y = df['target']

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = df.index[split:]

# Train models
with st.spinner("Training models..."):
    rf = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    xgb = XGBRegressor(n_estimators=n_trees, max_depth=max_depth,
                       learning_rate=learning_rate, random_state=42)
    xgb.fit(X_train, y_train)

    rf_preds   = rf.predict(X_test)
    xgb_preds  = xgb.predict(X_test)
    final_preds = (rf_weight * rf_preds) + (gb_weight * xgb_preds)

# Metrics
mae  = mean_absolute_error(y_test, final_preds)
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
r2   = r2_score(y_test, final_preds)
naive_mae = mean_absolute_error(y_test, X_test['close'].values)

# Display metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE",  f"${mae:,.0f}")
col2.metric("RMSE", f"${rmse:,.0f}")
col3.metric("R²",   f"{r2:.3f}")
col4.metric("vs Naive", f"${naive_mae - mae:,.0f}", delta_color="normal")

# Chart
st.subheader("Forecast vs Actual Price")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates_test, y_test.values, label="Actual Price", color="white", linewidth=1.5)
ax.plot(dates_test, final_preds,   label="Predicted",    color="orange", linewidth=1.5, linestyle="--")
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
ax.tick_params(colors="white")
ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
ax.legend(facecolor="#1a1a2e", labelcolor="white")
ax.set_ylabel("Price (USD)")
st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance (Random Forest)")
importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig2, ax2 = plt.subplots(figsize=(8, 5))
importance.plot(kind='barh', ax=ax2, color="orange")
ax2.set_facecolor("#1a1a2e")
fig2.patch.set_facecolor("#1a1a2e")
ax2.tick_params(colors="white")
st.pyplot(fig2)

st.caption("Data sourced from Yahoo Finance. Not financial advice.")