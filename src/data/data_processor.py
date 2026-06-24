from pathlib import Path

import pandas as pd


# tickers available
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]

# original columns
ORIGINAL_COLUMNS = ["Close", "High", "Low", "Open", "Volume"]

# number of previous days used to compute statistics
WINDOW_SIZES = [7, 30, 60]

# statistics computed
STATISTICS = ["min", "max", "mean", "std"]

# feature columns used by model
FEATURE_COLUMNS = ORIGINAL_COLUMNS + [
    f"{column}_{statistic}_{window_size}" 
    for column in ORIGINAL_COLUMNS 
    for window_size in WINDOW_SIZES
    for statistic in STATISTICS
]

# target columns to predict
TARGET_COLUMNS = ["Target_Close", "Target_High", "Target_Low", "Target_Open", "Target_Volume"]


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw stock data for the model.

    Args:
        data (pd.DataFrame): raw stock data.

    Returns:
        pd.DataFrame: processed stock data.

    """
    data = data.copy()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")

    previous_data = data[ORIGINAL_COLUMNS].shift(1) 
    for column in ORIGINAL_COLUMNS:
        for window_size in WINDOW_SIZES:
            rolling_values = previous_data[column].rolling(window=window_size)

            # calculate statistics of previous 7, 30 and 60 days for each row
            data[f"{column}_min_{window_size}"] = rolling_values.min()
            data[f"{column}_max_{window_size}"] = rolling_values.max()
            data[f"{column}_mean_{window_size}"] = rolling_values.mean()
            data[f"{column}_std_{window_size}"] = rolling_values.std()

    for column in ORIGINAL_COLUMNS:
        data[f"Target_{column}"] = data[column].shift(-1)

    data = data.dropna()
    data = data.reset_index(drop=True)

    columns = ["Date"] + FEATURE_COLUMNS + TARGET_COLUMNS
    data = data[columns]

    return data


def get_features_and_targets(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into features and targets.

    Args:
        data (pd.DataFrame): processed stock data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: input features and target values.

    """
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMNS]

    return X, y


def process_and_save_ticker(ticker: str) -> None:
    """
    Process and save data of a ticker.

    Args:
        ticker (str): stock ticker symbol.

    Returns:
        None

    """
    project_root = Path(__file__).resolve().parents[2]

    raw_data_path = project_root / "data" / "raw" / f"{ticker}.csv"
    data = pd.read_csv(raw_data_path)

    processed_data = process_data(data)
    processed_data_path = project_root / "data" / "processed" / f"{ticker}.csv"
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(processed_data_path, index=False)


def process_and_save_all_data() -> None:
    """
    Process and save data of all tickers.

    Returns:
        None

    """
    for ticker in TICKERS:
        process_and_save_ticker(ticker)


if __name__ == "__main__":
    process_and_save_all_data()

