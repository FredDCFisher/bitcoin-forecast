import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from data_fetch import fetch_bitcoin_data
from features import add_features

st.set_page_config(page_title="BTC/USD Forecaster", page_icon=None, layout="wide")

st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #0d1421;
        color: #e8eaed;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #111927;
        border-right: 1px solid #1e2d3d;
    }
    [data-testid="stSidebar"] * { color: #c5cdd8 !important; }

    #MainMenu, footer, header { visibility: hidden; }

    .page-header {
        padding: 1.5rem 0 0.25rem 0;
        border-bottom: 1px solid #1e2d3d;
        margin-bottom: 1.5rem;
    }
    .page-header h1 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
        letter-spacing: 0.02em;
    }
    .page-header p {
        font-size: 0.8rem;
        color: #6b7a8d;
        margin: 0.25rem 0 0 0;
    }

    .metric-card {
        background-color: #131d2e;
        border: 1px solid #1e2d3d;
        border-radius: 4px;
        padding: 1rem 1.25rem;
        height: 100%;
    }
    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7a8d;
        margin-bottom: 0.35rem;
    }
    .metric-sublabel {
        font-size: 0.65rem;
        color: #3d4f63;
        margin-top: 0.3rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        line-height: 1;
    }
    .metric-value.green  { color: #05c46b; }
    .metric-value.red    { color: #ff5e57; }
    .metric-value.amber  { color: #f5a623; }
    .metric-value.medium { font-size: 1.2rem; }

    .rsi-bar-track {
        background: #1e2d3d;
        border-radius: 2px;
        height: 4px;
        margin-top: 0.5rem;
        position: relative;
    }
    .rsi-bar-fill {
        height: 4px;
        border-radius: 2px;
    }
    .rsi-zones {
        display: flex;
        justify-content: space-between;
        font-size: 0.6rem;
        color: #3d4f63;
        margin-top: 0.2rem;
    }

    .section-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b7a8d;
        margin: 1.75rem 0 0.75rem 0;
        border-bottom: 1px solid #1e2d3d;
        padding-bottom: 0.4rem;
    }

    hr { border-color: #1e2d3d; }

    .disclaimer {
        font-size: 0.7rem;
        color: #3d4f63;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #1a2535;
    }

    label, .stSlider label { color: #c5cdd8 !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### Model Parameters")
n_trees       = st.sidebar.slider("Number of Trees", min_value=50, max_value=500, value=100, step=50)
max_depth     = st.sidebar.slider("Max Depth", min_value=3, max_value=10, value=5)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.30, value=0.10, step=0.01)
rf_weight     = st.sidebar.slider("Random Forest Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
gb_weight     = 1 - rf_weight
st.sidebar.markdown(
    f"<span style='font-size:0.78rem;color:#6b7a8d;'>XGBoost weight: "
    f"<b style='color:#c5cdd8'>{gb_weight:.1f}</b></span>",
    unsafe_allow_html=True
)

st.markdown("""
<div class="page-header">
    <h1>BTC / USD &nbsp; Forecaster</h1>
    <p>Machine learning price forecast &nbsp;·&nbsp; 5-year historical window &nbsp;·&nbsp; Ensemble model</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = fetch_bitcoin_data()
    df = add_features(df)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def fmt_volume(v):
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    if v >= 1e6:
        return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"


try:
    with st.spinner("Loading market data..."):
        df = load_data()

    feature_cols = [
        'close', 'ma7', 'ma30', 'lag_1', 'lag_3', 'lag_7',
        'daily_return', 'volatility_7', 'ma_crossover',
        'gold_change', 'sp500_change', 'ftse_change', 'eth_change'
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

    split      = int(len(df) * 0.8)
    X_train    = X.iloc[:split]
    X_test     = X.iloc[split:]
    y_train    = y.iloc[:split]
    y_test     = y.iloc[split:]
    dates_test = df.index[split:]

    with st.spinner("Training models..."):
        rf = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)

        xgb = XGBRegressor(n_estimators=n_trees, max_depth=max_depth,
                           learning_rate=learning_rate, random_state=42)
        xgb.fit(X_train, y_train)

        rf_preds    = rf.predict(X_test)
        xgb_preds   = xgb.predict(X_test)
        final_preds = rf_weight * rf_preds + gb_weight * xgb_preds

    mae         = mean_absolute_error(y_test, final_preds)
    rmse        = np.sqrt(mean_squared_error(y_test, final_preds))
    r2          = r2_score(y_test, final_preds)
    naive_mae   = mean_absolute_error(y_test, X_test['close'].values)
    improvement = naive_mae - mae

    # ── Market data ──────────────────────────────────────────────────────────
    current_price = df['close'].iloc[-1]
    high_24h      = df['high'].iloc[-1]   if 'high'   in df.columns else current_price
    low_24h       = df['low'].iloc[-1]    if 'low'    in df.columns else current_price
    volume_24h    = df['volume'].iloc[-1] if 'volume' in df.columns else 0

    rsi_series  = calc_rsi(df['close'])
    rsi_current = rsi_series.iloc[-1]

    if rsi_current >= 70:
        rsi_colour, rsi_label = "red",   "Overbought"
    elif rsi_current <= 30:
        rsi_colour, rsi_label = "green", "Oversold"
    else:
        rsi_colour, rsi_label = "amber", "Neutral"

    rsi_bar_pct   = f"{rsi_current:.0f}%"
    rsi_bar_color = {"red": "#ff5e57", "green": "#05c46b", "amber": "#f5a623"}[rsi_colour]

    # ── Forecast ─────────────────────────────────────────────────────────────
    latest_features   = X.iloc[[-1]]
    next_day_forecast = rf_weight * rf.predict(latest_features)[0] + gb_weight * xgb.predict(latest_features)[0]
    delta             = next_day_forecast - current_price
    delta_pct         = (delta / current_price) * 100
    delta_colour      = "green" if delta >= 0 else "red"
    delta_sign        = "+" if delta >= 0 else ""

    # ── Forecast Drift (maritime navigation concept) ──────────────────────────
    # In navigation, drift is how far a vessel has been pushed off its intended
    # course. Here: how much the model's rolling 30-day error has deviated from
    # its average — positive = model currently less accurate than usual.
    errors      = np.abs(y_test.values - final_preds)
    recent_mae  = np.mean(errors[-30:])
    drift_pct   = ((recent_mae - mae) / mae) * 100
    drift_sign  = "+" if drift_pct >= 0 else ""
    drift_colour = "red" if drift_pct > 5 else ("green" if drift_pct < -5 else "amber")

    # ── Layout ───────────────────────────────────────────────────────────────

    # Market Data
    st.markdown('<p class="section-title">Market Data</p>', unsafe_allow_html=True)
    d1, d2, d3, d4, d5 = st.columns(5)

    with d1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${current_price:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    with d2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">24h High</div>
            <div class="metric-value green">${high_24h:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    with d3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">24h Low</div>
            <div class="metric-value red">${low_24h:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    with d4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">24h Volume</div>
            <div class="metric-value medium">{fmt_volume(volume_24h)}</div>
        </div>""", unsafe_allow_html=True)

    with d5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value {rsi_colour}">{rsi_current:.1f}</div>
            <div class="rsi-bar-track">
                <div class="rsi-bar-fill" style="width:{rsi_current:.0f}%;background:{rsi_bar_color};"></div>
            </div>
            <div class="rsi-zones"><span>0</span><span>30</span><span>70</span><span>100</span></div>
            <div class="metric-sublabel">{rsi_label}</div>
        </div>""", unsafe_allow_html=True)

    # Forecast
    st.markdown('<p class="section-title">Forecast</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tomorrow Forecast</div>
            <div class="metric-value">${next_day_forecast:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    with f2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Expected Move</div>
            <div class="metric-value {delta_colour}">{delta_sign}${delta:,.0f} &nbsp;({delta_sign}{delta_pct:.2f}%)</div>
        </div>""", unsafe_allow_html=True)

    with f3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Forecast Drift</div>
            <div class="metric-value {drift_colour}">{drift_sign}{drift_pct:.1f}%</div>
            <div class="metric-sublabel">30-day error vs average &nbsp;·&nbsp; from maritime navigation</div>
        </div>""", unsafe_allow_html=True)

    # Model Performance
    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">${mae:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">${rmse:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">R²</div>
            <div class="metric-value">{r2:.3f}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        imp_colour = "green" if improvement > 0 else "red"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">vs Naive Baseline</div>
            <div class="metric-value {imp_colour}">${improvement:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    # Price chart
    BG      = "#0d1421"
    CARD_BG = "#111927"
    GRID    = "#1a2535"
    TEXT    = "#6b7a8d"
    GREEN   = "#05c46b"
    BLUE    = "#4d9de0"

    st.markdown('<p class="section-title">Forecast vs Actual</p>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD_BG)

    ax.plot(dates_test, y_test.values, color=GREEN, linewidth=1.4, label="Actual")
    ax.plot(dates_test, final_preds,   color=BLUE,  linewidth=1.4, linestyle="--", label="Forecast", alpha=0.85)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.grid(axis='x', color=GRID, linewidth=0.3)
    ax.set_ylabel("Price (USD)", color=TEXT, fontsize=8)
    ax.legend(facecolor=CARD_BG, labelcolor="#c5cdd8", fontsize=8,
              framealpha=1, edgecolor=GRID, loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)

    # Feature importance chart
    st.markdown('<p class="section-title">Signal Importance &nbsp; (Random Forest)</p>', unsafe_allow_html=True)

    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    clean_labels = {
        'close': 'Close Price', 'ma7': 'MA 7d', 'ma30': 'MA 30d',
        'lag_1': 'Lag 1d', 'lag_3': 'Lag 3d', 'lag_7': 'Lag 7d',
        'daily_return': 'Daily Return', 'volatility_7': 'Volatility 7d',
        'ma_crossover': 'MA Crossover', 'gold_change': 'Gold',
        'sp500_change': 'S&P 500', 'ftse_change': 'FTSE 100', 'eth_change': 'Ethereum'
    }
    importance.index = [clean_labels.get(i, i) for i in importance.index]

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    fig2.patch.set_facecolor(BG)
    ax2.set_facecolor(CARD_BG)

    ax2.barh(importance.index, importance.values, color=BLUE, height=0.55)
    ax2.tick_params(colors=TEXT, labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID)
    ax2.grid(axis='x', color=GRID, linewidth=0.6)
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    fig2.tight_layout()
    st.pyplot(fig2)

    st.markdown(
        '<p class="disclaimer">Data sourced from Yahoo Finance via yfinance. '
        'Past performance does not guarantee future results. Not financial advice.</p>',
        unsafe_allow_html=True
    )

except Exception as e:
    st.error(str(e))
