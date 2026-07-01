from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.config import TICKERS, FEATURE_COLUMNS, TARGET_COLUMNS, TEST_SIZE, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, TARGET_DAYS, VALIDATION_SIZE, ALPHA_VALUES


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
        ticker (str): ticker symbol.
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


def _find_best_alpha(X_train_validation: pd.DataFrame, y_train_validation: pd.DataFrame) -> float:
    """
    Find the best parameter alpha using a validation set.

    Args:
        X_train_validation (pd.DataFrame): features used for training and validation.
        y_train_validation (pd.DataFrame): targets used for training and validation.

    Returns:
        float: best alpha value.
    """
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size=VALIDATION_SIZE, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)

    best_alpha = 0.1
    best_mae = None

    for alpha in ALPHA_VALUES:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        y_validation_pred = model.predict(X_validation)

        mae = mean_absolute_error(y_validation, y_validation_pred)
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    print(f"Best alpha: {best_alpha}")

    return best_alpha


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

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(X,y,test_size=TEST_SIZE,shuffle=False)

    best_alpha = _find_best_alpha(X_train_validation, y_train_validation)

    scaler = StandardScaler()
    X_train_validation = scaler.fit_transform(X_train_validation)
    X_test = scaler.transform(X_test)

    model = Ridge(alpha=best_alpha)
    model.fit(X_train_validation, y_train_validation)

    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=TARGET_COLUMNS, index=y_test.index)

    # multioutput="raw_values" -> return a different value for each target column, instead of calculating the average of them
    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")  # penalize significant errors
    rmse = mse ** 0.5

    metrics = pd.DataFrame({"target": TARGET_COLUMNS, "mae": mae, "rmse": rmse})

    # DataFrame used for the evaluation model graphic
    predictions = pd.DataFrame()
    predictions["Date"] = data.loc[y_test.index, "Date"]
    predictions["Current_Average_Price"] = data.loc[y_test.index, "avg_current"]

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
    if ticker not in TICKERS:
        print(f"Invalid ticker: {ticker}")
        print(f"Available tickers: {TICKERS}")
        sys.exit(1)

    train_and_evaluate_model(ticker)

