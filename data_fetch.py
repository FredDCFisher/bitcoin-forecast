import yfinance as yf
import pandas as pd

def download_close(ticker):
    df = yf.download(ticker, period='5y', interval='1d', auto_adjust=True, progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df[['Close']]

def fetch_bitcoin_data():
    print("Fetching Bitcoin data...")
    btc = yf.download('BTC-USD', period='5y', interval='1d', auto_adjust=True, progress=False)
    btc.columns = [col[0] if isinstance(col, tuple) else col for col in btc.columns]
    btc = btc[['Close', 'Open', 'High', 'Low', 'Volume']]
    btc.columns = ['close', 'open', 'high', 'low', 'volume']

    # Moving averages
    btc['ma7']  = btc['close'].rolling(window=7).mean()
    btc['ma30'] = btc['close'].rolling(window=30).mean()

    # External market data
    print("Fetching external market data...")
    gold  = download_close('GC=F').rename(columns={'Close': 'gold'})
    sp500 = download_close('^GSPC').rename(columns={'Close': 'sp500'})
    ftse  = download_close('^FTSE').rename(columns={'Close': 'ftse'})
    eth   = download_close('ETH-USD').rename(columns={'Close': 'eth'})

    # Combine everything into one table by date
    df = btc.join([gold, sp500, ftse, eth], how='left')
    df.dropna(subset=['close'], inplace=True)

    print(f"Done! {len(df)} days of data loaded.")
    print(df.tail())
    return df

if __name__ == "__main__":
    df = fetch_bitcoin_data()