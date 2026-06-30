from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.config import FEATURE_COLUMNS, MODELS_DIR, PREDICTION_DAYS, PREDICTIONS_DIR, PREVIOUS_DAYS, RAW_DATA_DIR, TICKERS, WINDOW_SIZES


def _load_raw_data(ticker: str) -> pd.DataFrame:
    """
    Load raw data of a ticker.

    Args:
        ticker (str): ticker symbol.

    Returns:
        pd.DataFrame: raw ticker data.
    """
    file_path = RAW_DATA_DIR / f"{ticker}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    return pd.read_csv(file_path)


def _load_model_and_scaler(ticker: str) -> tuple:
    """
    Load trained model and scaler.

    Args:
        ticker (str): ticker symbol.

    Returns:
        tuple: trained model and scaler.
    """
    model_path = MODELS_DIR / f"{ticker}_ridge_regression.joblib"
    scaler_path = MODELS_DIR / f"{ticker}_standard_scaler.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


def _build_feature_row(data: pd.DataFrame) -> dict:
    """
    Build one feature row using the last available stock data.

    Args:
        data (pd.DataFrame): raw data.

    Returns:
        dict: feature row used by the model.
    """
    data = data.copy()

    # make sure data is ordered by date
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")

    # create column with the avg price for each day
    data["avg_current"] = (data["High"] + data["Low"]) / 2

    # min number of values needed to predict avg prices for next 7 days
    min_required_values = max(PREVIOUS_DAYS) + 1
    # filter dataset by considering only needed rows
    data = data.tail(min_required_values)
    avg_values = data["avg_current"].tolist()

    features = {}

    # avg price of current day is the last value of the list
    features["avg_current"] = avg_values[-1]

    # create columns avg_prev_1, avg_prev_2, ... for each previous day that is considered by the model
    for prev in PREVIOUS_DAYS:
        features[f"avg_prev_{prev}"] = avg_values[-1 - prev]

    # previous avg values (current day excluded)
    previous_values = avg_values[:-1]

    # calculate statistics for previous 7 days
    for window_size in WINDOW_SIZES:
        window_values = previous_values[-window_size:]

        features[f"stat_min_{window_size}"] = min(window_values)
        features[f"stat_max_{window_size}"] = max(window_values)
        features[f"stat_mean_{window_size}"] = sum(window_values) / len(window_values)
        features[f"stat_std_{window_size}"] = pd.Series(window_values).std()

    return features


def predict_next_7_days(ticker: str) -> pd.DataFrame:
    """
    Predict the next 7 average prices of a ticker.

    Args:
        ticker (str): ticker symbol.

    Returns:
        pd.DataFrame: predictions for the next 7 days.
    """
    data = _load_raw_data(ticker)
    model, scaler = _load_model_and_scaler(ticker)

    feature_row = _build_feature_row(data)

    X = pd.DataFrame([feature_row])
    X = X[FEATURE_COLUMNS]

    X_scaled = scaler.transform(X)

    # the return is [[...]]
    y_pred = model.predict(X_scaled)[0]

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")
    last_date = data["Date"].iloc[-1]

    # find business day
    prediction_dates = pd.bdate_range(
        start=last_date + pd.offsets.BDay(1),
        periods=PREDICTION_DAYS
    )

    predictions = pd.DataFrame({
        "Day": list(range(1, PREDICTION_DAYS + 1)),
        "Date": prediction_dates,
        "Predicted_Average_Price": y_pred
    })

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PREDICTIONS_DIR / f"{ticker}_predictions.csv"
    predictions.to_csv(output_path, index=False)

    return predictions


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: <ticker> required as argument")
        print(f"Available tickers: {TICKERS}")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    if ticker not in TICKERS:
        print(f"Invalid ticker: {ticker}")
        print(f"Available tickers: {TICKERS}")
        sys.exit(1)

    predictions = predict_next_7_days(ticker)

