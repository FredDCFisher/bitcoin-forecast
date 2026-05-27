import pandas as pd
from data_fetch import fetch_bitcoin_data

def add_features(df):
    print("Building features...")

    # Lag features — what was the price 1, 3 and 7 days ago?
    df['lag_1'] = df['close'].shift(1)
    df['lag_3'] = df['close'].shift(3)
    df['lag_7'] = df['close'].shift(7)

    # Daily % change — how much did price move today?
    df['daily_return'] = df['close'].pct_change() * 100

    # Volatility — how jumpy has price been over last 7 days?
    df['volatility_7'] = df['daily_return'].rolling(window=7).std()

    # MA crossover signal — is short term average above long term?
    # 1 = bullish (possible uptrend), 0 = bearish (possible downtrend)
    df['ma_crossover'] = (df['ma7'] > df['ma30']).astype(int)

    # External market daily % changes
    df['gold_change']  = df['gold'].pct_change() * 100
    df['sp500_change'] = df['sp500'].pct_change() * 100
    df['ftse_change']  = df['ftse'].pct_change() * 100
    df['eth_change']   = df['eth'].pct_change() * 100

    # Drop rows with missing values (first 30 days will have gaps)
    df.dropna(inplace=True)

    print(f"Done! {len(df)} rows, {len(df.columns)} features")
    print(df.tail(3))
    return df

if __name__ == "__main__":
    df = fetch_bitcoin_data()
    df = add_features(df)