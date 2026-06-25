from pathlib import Path

import pandas as pd
import yfinance as yf

# tickers available
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]

# period considered (10 years)
START_DATE = "2006-06-01"
END_DATE = "2026-06-01" # not included in the period


def _load_data(ticker):
    """
    Download historical stock data of a ticker.

    Args:
        ticker (str): stock ticker symbol.
    
    Returns:
        pd.DataFrame: historical stock data with six columns: Date, Open, High, Low, Close and Volume.

    """
    # download historical data:
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

    # columns are: (Close, <ticker name>), (High, <ticker name>), (Low, <ticker name>), (Open, <ticker name>), (Volume, <ticker name>)
    # data of each ticker is saved on a different file, so the <ticker_name> isn't needed
    data.columns = data.columns.get_level_values(0)
    data.columns.name = None # Remove the old name of the group of columns (price)
    data = data.reset_index()

    return data


def _save_data(ticker, data):
    """
    Save the stock data of a ticker into a CSV file.

    Args:
        ticker (str): stock ticker symbol.
        data (pd.DataFrame): stock data to save.
    """
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    # parents=True: create intermediate directories if they don't exist
    # exist_ok=True: don't show any error if the directory already exists

    output_path = output_dir / f"{ticker}.csv"
    data.to_csv(output_path, index=False)
    # index=False: don't save the index (0,1,2...) as a new column

    print(f"Data for {ticker} saved to {output_path}")


if __name__ == "__main__":
    for ticker in TICKERS:
        print(f"Loading data for {ticker}...")
        data = _load_data(ticker)
        _save_data(ticker, data)

