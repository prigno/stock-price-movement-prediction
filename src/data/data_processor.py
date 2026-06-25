from pathlib import Path
import sys

import pandas as pd


# available tickers
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]

# number of previous days for which avg is computed
PREVIOUS_DAYS = list(range(1, 61))

# number of previous days used to compute statistics
WINDOW_SIZES = [7, 14, 30, 60]

# statistics computed
STATISTICS = ["min", "max", "mean", "std"]

# feature columns used by model 
FEATURE_COLUMNS = (
    ["avg_current"] + 
    [f"avg_prev_{prev}" for prev in PREVIOUS_DAYS] + 
    [f"stat_{statistic}_{window_size}" for window_size in WINDOW_SIZES for statistic in STATISTICS]
)

TARGET_DAYS = list(range(1,8))

# target column to predict
TARGET_COLUMN = [f"Target_Average_Price_{day}" for day in TARGET_DAYS]


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

    data["avg_current"] = (data["High"] + data["Low"]) / 2 # create column with avg price of the current day

    for prev in PREVIOUS_DAYS:
        # create columns witch the avg prices of the previous 30 days, one for each column
        data[f"avg_prev_{prev}"] = data["avg_current"].shift(prev)

    previous_data = data["avg_current"].shift(1) # Series containing the previous 30 avg prices (today excluded)
    for window_size in WINDOW_SIZES:
        # sliding window of the previouse 7, 14, 30 or 60 days
        rolling_values = previous_data.rolling(window=window_size) 

        # calculate statistics of previous 7, 14, 30 or 60 days for each row
        data[f"stat_min_{window_size}"] = rolling_values.min()
        data[f"stat_max_{window_size}"] = rolling_values.max()
        data[f"stat_mean_{window_size}"] = rolling_values.mean()
        data[f"stat_std_{window_size}"] = rolling_values.std()

    for day in TARGET_DAYS:
        # the model must predict for each row the avr price for the next 7 days
        data[f"Target_Average_Price_{day}"] = data["avg_current"].shift(-day)

    data = data.dropna()
    data = data.reset_index(drop=True) # adjust indexes because some of the first rows
                                       # and some of the last rows have been removed from dropna()
    # columns of the processed dataset
    columns = ["Date"] + FEATURE_COLUMNS + TARGET_COLUMN
    data = data[columns]

    return data


def _process_and_save_ticker(ticker: str) -> None:
    """
    Process and save data of a ticker.

    Args:
        ticker (str): stock ticker symbol.
    """
    project_root = Path(__file__).resolve().parents[2]

    raw_data_path = project_root / "data" / "raw" / f"{ticker}.csv"
    data = pd.read_csv(raw_data_path)

    processed_data = _process_data(data)
    processed_data_path = project_root / "data" / "processed" / f"{ticker}.csv"
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    processed_data.to_csv(processed_data_path, index=False)


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
        _process_and_save_ticker(ticker)
    else:
        for ticker in TICKERS:
            _process_and_save_ticker(ticker)

