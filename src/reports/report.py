from pathlib import Path
import sys

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.config import REPORTS_DIR, TICKERS
from src.data.data_loader import load_and_save_ticker
from src.data.data_processor import process_and_save_ticker
from src.models.ridge_regression import train_and_evaluate_model


def _load_metrics() -> pd.DataFrame:
    """
    Load Ridge Regression metrics and calculate the mean metrics over all tickers.

    Returns:
        pd.DataFrame: mean model evaluation metrics over all tickers.
    """
    dataframes = []

    for ticker in TICKERS:
        load_and_save_ticker(ticker)
        process_and_save_ticker(ticker)
        train_and_evaluate_model(ticker)

        file_path = REPORTS_DIR / f"{ticker}_ridge_regression_metrics.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {file_path}")

        df = pd.read_csv(file_path)
        
        dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)

    mean_metrics = data.groupby("target", as_index=False)[["mae", "rmse"]].mean()
    print(mean_metrics)

    output_path = REPORTS_DIR / "ridge_regression_mean_metrics.csv"
    mean_metrics.to_csv(output_path, index=False)
    
    return mean_metrics


def metrics_plot() -> Path:
    """
    Create Ridge Regression MAE and RMSE plot.

    Returns:
        Path: path of the generated image.
    """
    data = _load_metrics()
    data["target_day"] = data["target"].str.extract(r"(\d+)$").astype(int)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = REPORTS_DIR / f"ridge_regression_metrics.svg"

    x_values = range(len(data))
    x_labels = [f"Day {day}" for day in data["target_day"]]
    bar_width = 0.35

    plt.figure(figsize=(1200, 600, "px"))

    plt.bar([x - bar_width / 2 for x in x_values],data["mae"], width=bar_width, label="MAE")
    plt.bar([x + bar_width / 2 for x in x_values], data["rmse"], width=bar_width, label="RMSE")
    plt.title(f"Mae and RMSE")
    plt.xlabel("Target day")
    plt.ylabel("Error ($)")
    plt.xticks(ticks=list(x_values), labels=x_labels)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    print(f"MAE and RMSE plot for saved to {image_path}")

    return image_path


if __name__ == "__main__":
    try:
        metrics_plot()
    except (FileNotFoundError, ValueError) as error:
        print(error)
        sys.exit(1)
