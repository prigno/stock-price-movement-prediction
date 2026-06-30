from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.config import TICKERS, FEATURE_COLUMNS, TARGET_COLUMNS, TEST_SIZE, PROCESSED_DATA_DIR, REPORTS_DIR


def _load_processed_data(ticker: str) -> pd.DataFrame:
    """
    Load processed data of a ticker.

    Args:
        ticker (str): ticker symbol.

    Returns:
        pd.DataFrame: processed ticker data.
    """
    file_path = PROCESSED_DATA_DIR / f"{ticker}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {file_path}")

    return pd.read_csv(file_path)


def _save_report(ticker: str, metrics: pd.DataFrame) -> None:
    """
    Save evaluation metrics.

    Args:
        ticker (str): ticker symbol.
        metrics (pd.DataFrame): evaluation metrics.
    """
    output_dir = REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"{ticker}_polynomial_regression_metrics.csv"

    metrics.to_csv(report_path, index=False)


def train_and_evaluate_model(ticker: str) -> pd.DataFrame:
    """
    Train and evaluate the model for a ticker.

    Args:
        ticker (str): ticker symbol.

    Returns:
        pd.DataFrame: evaluation metrics.
    """
    data = _load_processed_data(ticker)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

    polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
    X_train = polynomial_features.fit_transform(X_train)
    X_test = polynomial_features.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=TARGET_COLUMNS, index=y_test.index)

    # multioutput="raw_values" -> return a different value for each target column, instead of calculating the average of them
    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")  # penalize significant errors
    rmse = mse ** 0.5

    metrics = pd.DataFrame({"target": TARGET_COLUMNS, "mae": mae, "rmse": rmse})
    _save_report(ticker, metrics)

    print(f"Ticker: {ticker}")
    print(metrics)

    return metrics


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

    train_and_evaluate_model(ticker)

