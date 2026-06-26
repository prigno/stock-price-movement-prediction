from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


from src.config import TICKERS, FEATURE_COLUMNS, TARGET_COLUMNS, TEST_SIZE, ALPHA, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, TARGET_DAYS


def _load_processed_data(ticker: str) -> pd.DataFrame:
    """
    Load processed data of a ticker.

    Args:
        ticker (str): stock ticker symbol.

    Returns:
        pd.DataFrame: processed ticker data.
    """
    file_path = PROCESSED_DATA_DIR / f"{ticker}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {file_path}")

    return pd.read_csv(file_path)


def _save_model(ticker: str, model: Ridge, scaler: StandardScaler) -> None:
    """
    Save the trained model.

    Args:
        ticker (str): stock ticker symbol.
        model (Ridge): trained Ridge Regression model.
    """
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{ticker}_ridge_regression.joblib"
    scaler_path = model_dir / f"{ticker}_standard_scaler.joblib"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def _save_report(ticker: str, metrics: pd.DataFrame) -> None:
    """
    Save evaluation metrics.

    Args:
        ticker (str): stock ticker symbol.
        metrics (pd.DataFrame): evaluation metrics.
    """
    output_dir = REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"{ticker}_ridge_regression_metrics.csv"

    metrics.to_csv(report_path, index=False)


def _save_test_predictions(ticker: str, test_predictions: pd.DataFrame) -> None:
    """
    Save testing set predictions.

    Args:
        ticker (str): ticker symbol.
        test_predictions (pd.DataFrame): actual and predicted values on the test set.
    """
    output_dir = REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{ticker}_ridge_regression_test_predictions.csv"

    test_predictions.to_csv(report_path, index=False)


def train_and_evaluate_model(ticker: str) -> pd.DataFrame:
    """
    Train and evaluate the Ridge Regression model for a ticker.

    Args:
        ticker (str): stock ticker symbol.

    Returns:
        pd.DataFrame: evaluation metrics.
    """
    data = _load_processed_data(ticker)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Ridge(alpha=ALPHA)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=TARGET_COLUMNS, index=y_test.index)

    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred, multioutput="raw_values")

    metrics = pd.DataFrame({"target": TARGET_COLUMNS, "mae": mae, "rmse": rmse, "r2": r2})

    # DataFrame used for the evaluation model graphic
    predictions = pd.DataFrame()
    predictions["Date"] = data.loc[y_test.index, "Date"]

    for day in TARGET_DAYS:
        predictions[f"Actual_Average_Price_{day}"] = y_test[f"Target_Average_Price_{day}"]
        predictions[f"Predicted_Average_Price_{day}"] = y_pred[f"Target_Average_Price_{day}"]
        
        predictions[f"Absolute_Error_{day}"] = (
            predictions[f"Actual_Average_Price_{day}"] -
            predictions[f"Predicted_Average_Price_{day}"]
        ).abs()

    _save_model(ticker, model, scaler)
    _save_report(ticker, metrics)
    _save_test_predictions(ticker, predictions)

    print(f"Ticker: {ticker}")
    print(metrics)

    return metrics


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
        train_and_evaluate_model(ticker)
    else:
        for t in TICKERS:
            train_and_evaluate_model(t)

