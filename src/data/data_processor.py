from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.config import TICKERS, PREVIOUS_DAYS, WINDOW_SIZES, FEATURE_COLUMNS, TARGET_DAYS, TARGET_COLUMNS, RAW_DATA_DIR, PROCESSED_DATA_DIR


def _process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw stock data for the model.

    Args:
        data (pd.DataFrame): raw stock data.

    Returns:
        pd.DataFrame: processed stock data.
    """
    data = data.copy() # don't modify the original df

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date") # make sure dates are ordered

    data["avg_current"] = (data["High"] + data["Low"]) / 2 # create column with avg price of each day

    for prev in PREVIOUS_DAYS:
        # create columns witch the avg prices of the previous days, one for each column
        data[f"avg_prev_{prev}"] = data["avg_current"].shift(prev)

    previous_data = data["avg_current"].shift(1) # Series containing the previous avg prices (current day excluded)
    for window_size in WINDOW_SIZES:
        # sliding window of the previouse days
        rolling_values = previous_data.rolling(window=window_size) 

        # calculate statistics of previous 7 days for each row
        data[f"stat_min_{window_size}"] = rolling_values.min()
        data[f"stat_max_{window_size}"] = rolling_values.max()
        data[f"stat_mean_{window_size}"] = rolling_values.mean()
        data[f"stat_std_{window_size}"] = rolling_values.std()

    for day in TARGET_DAYS:
        # the model try to predict for each row the avg price for the next days
        data[f"Target_Average_Price_{day}"] = data["avg_current"].shift(-day)

    data = data.dropna()
    data = data.reset_index(drop=True)

    # columns of the processed dataset
    columns = ["Date"] + FEATURE_COLUMNS + TARGET_COLUMNS

    return data[columns]


def process_and_save_ticker(ticker: str) -> None:
    """
    Process and save data of a ticker.

    Args:
        ticker (str): ticker symbol.
    """
    
    raw_data_path = RAW_DATA_DIR / f"{ticker}.csv"
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
    data = pd.read_csv(raw_data_path)

    processed_data = _process_data(data)

    output_dir = PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}.csv"

    processed_data.to_csv(output_path, index=False)
    print(f"Processed data for {ticker} saved to {output_path}")


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

    process_and_save_ticker(ticker)

