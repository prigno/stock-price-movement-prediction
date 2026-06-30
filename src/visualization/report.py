from pathlib import Path
import sys

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.config import REPORTS_DIR, TICKERS

METRIC_COLUMNS = ["mae", "rmse"]


def _load_metrics(ticker: str) -> pd.DataFrame:
    """
    Load Ridge Regression metrics of a ticker.

    Args:
        ticker (str): ticker symbol.

    Returns:
        pd.DataFrame: model evaluation metrics.
    """
    file_path = REPORTS_DIR / f"{ticker}_ridge_regression_metrics.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")

    data = pd.read_csv(file_path)

    return data


def _add_target_day(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add the column 'day'.

    Args:
        data (pd.DataFrame): metrics data.

    Returns:
        pd.DataFrame: metrics data with target day.
    """
    data = data.copy()

    # extract the final number from a column like "Target_Average_Price_1"
    data["target_day"] = data["target"].str.extract(r"(\d+)$").astype(int)

    return data


def metrics_plot(ticker: str) -> Path:
    """
    Create Ridge Regression MAE and RMSE plot.

    Args:
        ticker (str): ticker symbol.

    Returns:
        Path: path of the generated image.
    """
    data = _load_metrics(ticker)
    data = _add_target_day(data)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = REPORTS_DIR / f"{ticker}_ridge_regression_metrics.svg"

    x_values = range(len(data))
    x_labels = [f"Day {day}" for day in data["target_day"]]
    bar_width = 0.35

    plt.figure(figsize=(1200, 600, "px"))

    plt.bar([x - bar_width / 2 for x in x_values],data["mae"], width=bar_width, label="MAE")
    plt.bar([x + bar_width / 2 for x in x_values], data["rmse"], width=bar_width, label="RMSE")
    plt.title(f"{ticker} - Mae and RMSE")
    plt.xlabel("Target day")
    plt.ylabel("Error ($)")
    plt.xticks(ticks=list(x_values), labels=x_labels)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    print(f"MAE and RMSE plot for {ticker} saved to {image_path}")

    return image_path


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

    try:
        metrics_plot(ticker)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        sys.exit(1)
