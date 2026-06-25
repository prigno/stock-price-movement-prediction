from pathlib import Path
import sys

import pandas as pd
import yfinance as yf

# tickers available
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]

# period considered
START_DATE = "2000-01-01"
END_DATE = "2026-06-01"


def _load_data(ticker, start_date=START_DATE, end_date=END_DATE):
    """
    Download historical stock data of a ticker.

    Args:
        ticker (str): stock ticker symbol.
        start_date (str): first day of the period (included)
        end_date (str): last day of the periodo (excluded)
    
    Returns:
        pd.DataFrame: historical stock data with six columns: Date, Open, High, Low, Close and Volume.

    """
    # download historical data:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

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


def load_and_save_ticker(ticker, start_date=START_DATE, end_date=END_DATE):
    """
    Download and save stock data of a ticker.

    Args:
        ticker (str): stock ticker symbol.
        start_date (str): first date included in the period.
        end_date (str): last date not included in the period.
    """
    print(f"Loading data for {ticker}...")
    data = _load_data(ticker, start_date=start_date, end_date=end_date)
    _save_data(ticker, data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: <ticker> required as argument")
        print(f"Available tickrs:e {TICKERS}")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    if ticker not in TICKERS and ticker != "ALL_TICKERS":
        print(f"Invalid ticker: {ticker}")
        print(f"Available tickers: {TICKERS}")
        sys.exit(1)

    if ticker != "ALL_TICKERS":
        load_and_save_ticker(ticker)
    else:
        for t in TICKERS:
            load_and_save_ticker(t)
            

