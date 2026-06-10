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

# Sidebar
st.sidebar.header("Model Parameters")

n_trees = st.sidebar.slider(
    "Number of Trees",
    min_value=50,
    max_value=500,
    value=100,
    step=50
)

max_depth = st.sidebar.slider(
    "Max Depth",
    min_value=3,
    max_value=10,
    value=5
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.01,
    max_value=0.30,
    value=0.10,
    step=0.01
)

rf_weight = st.sidebar.slider(
    "Random Forest Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

gb_weight = 1 - rf_weight

st.sidebar.markdown(f"XGBoost weight: **{gb_weight:.1f}**")


@st.cache_data
def load_data():
    df = fetch_bitcoin_data()
    df = add_features(df)

    df.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in df.columns
    ]

    return df


try:
    with st.spinner("Loading data..."):
        df = load_data()

    feature_cols = [
        'close',
        'ma7',
        'ma30',
        'lag_1',
        'lag_3',
        'lag_7',
        'daily_return',
        'volatility_7',
        'ma_crossover',
        'gold_change',
        'sp500_change',
        'ftse_change',
        'eth_change'
    ]

    df['target'] = df['close'].shift(-1)

    missing_cols = [c for c in feature_cols if c not in df.columns]

    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    X = df[feature_cols].astype(float)
    y = df['target'].astype(float)

    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    dates_test = df.index[split:]

    with st.spinner("Training models..."):

        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            random_state=42
        )

        rf.fit(X_train, y_train)

        xgb = XGBRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )

        xgb.fit(X_train, y_train)

        rf_preds = rf.predict(X_test)
        xgb_preds = xgb.predict(X_test)

        final_preds = (
            rf_weight * rf_preds +
            gb_weight * xgb_preds
        )

    mae = mean_absolute_error(y_test, final_preds)
    rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    r2 = r2_score(y_test, final_preds)

    naive_mae = mean_absolute_error(
        y_test,
        X_test['close'].values
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE", f"${mae:,.0f}")
    c2.metric("RMSE", f"${rmse:,.0f}")
    c3.metric("R²", f"{r2:.3f}")
    c4.metric("vs Naive", f"${naive_mae - mae:,.0f}")

     # Tomorrow forecast
    latest_features = X.iloc[[-1]]

    rf_next = rf.predict(latest_features)[0]
    xgb_next = xgb.predict(latest_features)[0]

    next_day_forecast = (
        rf_weight * rf_next +
        gb_weight * xgb_next
    )

    st.markdown("---")

    forecast_col, current_col = st.columns(2)

    forecast_col.metric(
        "🔮 Tomorrow Forecast",
        f"${next_day_forecast:,.0f}"
    )

    current_col.metric(
        "📈 Current Price",
        f"${df['close'].iloc[-1]:,.0f}"
    )
    st.subheader("Forecast vs Actual Price")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        dates_test,
        y_test.values,
        label="Actual Price",
        color="white",
        linewidth=1.5
    )

    ax.plot(
        dates_test,
        final_preds,
        label="Predicted",
        color="orange",
        linewidth=1.5,
        linestyle="--"
    )

    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.set_ylabel("Price (USD)", color="white")

    st.pyplot(fig)

    st.subheader("Feature Importance (Random Forest)")

    importance = pd.Series(
        rf.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=True)

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    importance.plot(
        kind="barh",
        ax=ax2,
        color="orange"
    )

    ax2.set_facecolor("#1a1a2e")
    fig2.patch.set_facecolor("#1a1a2e")
    ax2.tick_params(colors="white")

    st.pyplot(fig2)

    st.caption(
        "Data sourced from Yahoo Finance. Not financial advice."
    )
except Exception as e:
    st.error(str(e))