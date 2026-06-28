from pathlib import Path
import sys

import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.config import TICKERS, START_DATE, END_DATE, RAW_DATA_DIR


def _load_data(ticker, start_date=START_DATE, end_date=END_DATE):
    """
    Download historical stock data of a ticker.

    Args:
        ticker (str): ticker symbol.
        start_date (str): first day of the period (included)
        end_date (str): last day of the periodo (excluded)
    
    Returns:
        pd.DataFrame: historical stock data.

    """
    # download historical data:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # columns are: (Close, <ticker name>), (High, <ticker name>), (Low, <ticker name>), (Open, <ticker name>), (Volume, <ticker name>)
    # data of each ticker is saved on a different file, so the <ticker_name> isn't needed
    data.columns = data.columns.get_level_values(0)
    data.columns.name = None # Remove the name of the group of columns (price)

    data = data.dropna()
    data = data.reset_index()

    return data


def _save_data(ticker, data):
    """
    Save the stock data of a ticker into a CSV file.

    Args:
        ticker (str): ticker symbol.
        data (pd.DataFrame): stock data to save.
    """
    output_dir = RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{ticker}.csv"
    # index=False: don't save the index (0,1,2...) as a new column
    data.to_csv(output_path, index=False)

    print(f"Data for {ticker} saved to {output_path}")


def load_and_save_ticker(ticker, start_date=START_DATE, end_date=END_DATE):
    """
    Download and save stock data of a ticker.

    Args:
        ticker (str): ticker symbol.
        start_date (str): first date of the period.
        end_date (str): last date of the period (excluded).
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
    if ticker not in TICKERS:
        print(f"Invalid ticker: {ticker}")
        print(f"Available tickers: {TICKERS}")
        sys.exit(1)

    load_and_save_ticker(ticker)
            
