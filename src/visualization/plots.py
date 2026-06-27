from pathlib import Path
import sys

import matplotlib
matplotlib.use("svg") # use a non interactive backend because want to save the files only

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.config import STATIC_IMAGES_DIR, REPORTS_DIR, PREDICTIONS_DIR, TARGET_DAYS


def model_evaluation_plot(ticker: str, day: int) -> str:
    """
    Create model evaluation plot.

    Args:
        ticker (str): ticker symbol.
        day (int): target day.

    Returns:
        str: relative path of the generated image.
    """
    predictions_path = REPORTS_DIR / f"{ticker}_ridge_regression_test_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Test predictions file not found: {predictions_path}")

    data = pd.read_csv(predictions_path)
    data["Date"] = pd.to_datetime(data["Date"])

    image_name = f"{ticker}_model_evaluation_{day}.svg"
    STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = STATIC_IMAGES_DIR / image_name

    plt.figure(figsize=(1200, 600, "px"))
    plt.plot(data["Date"], data[f"Actual_Average_Price_{day}"], label="Real values")
    plt.plot(data["Date"], data[f"Predicted_Average_Price_{day}"], label="Predicted values")
    plt.title(f"{ticker} Model Evaluation - Day {day}")
    plt.xlabel("Date")
    plt.ylabel("Average Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    return f"images/{image_name}"


def error_plot(ticker: str, day: int) -> str:
    """
    Create absolute error plot.

    Args:
        ticker (str): stock ticker symbol.
        day (int): target day.

    Returns:
        str: relative path of the generated image.
    """
    predictions_path = REPORTS_DIR / f"{ticker}_ridge_regression_test_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Test predictions file not found: {predictions_path}")

    data = pd.read_csv(predictions_path)
    data["Date"] = pd.to_datetime(data["Date"])

    image_name = f"{ticker}_absolute_error_{day}.svg"
    STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = STATIC_IMAGES_DIR / image_name

    plt.figure(figsize=(1200, 600, "px"))
    #plt.plot(data["Date"], data[f"Absolute_Error_{day}"])
    sns.histplot(data=data, x=f"Absolute_Error_{day}", bins=30, kde=True)
    mean_error = data[f"Absolute_Error_{day}"].mean()
    plt.axvline(mean_error, linestyle="--", label=f"Mean error: {mean_error:.2f}")
    plt.title(f"{ticker} Absolute Error - Day {day}")
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    return f"images/{image_name}"


def prediction_plot(ticker: str) -> str:
    """
    Create next 7 days prediction plot.

    Args:
        ticker (str): ticker symbol.

    Returns:
        str: relative path of the generated image.
    """
    predictions_path = PREDICTIONS_DIR / f"{ticker}_ridge_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")

    data = pd.read_csv(predictions_path)
    data["Date"] = pd.to_datetime(data["Date"])

    image_name = f"{ticker}_next_7_days_predictions.svg"
    STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = STATIC_IMAGES_DIR / image_name

    plt.figure(figsize=(1200, 600, "px"))
    plt.plot(data["Date"], data["Predicted_Average_Price"], marker="o")
    plt.xticks(ticks=data["Date"], labels=data["Date"].dt.strftime("%Y-%m-%d"), rotation=30)
    plt.title(f"{ticker} Next 7 Business Days Predictions")
    plt.xlabel("Date")
    plt.ylabel("Predicted Average Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    return f"images/{image_name}"


def all_plots(ticker: str) -> dict:
    """
    Create all plots for the results page.

    Args:
        ticker (str): stock ticker symbol.

    Returns:
        dict: generated image paths.
    """
    image_paths = {
        "model_evaluation": {},
        "absolute_error": {},
        "predictions": prediction_plot(ticker)
    }

    for day in TARGET_DAYS:
        image_paths["model_evaluation"][day] = model_evaluation_plot(ticker, day)
        image_paths["absolute_error"][day] = error_plot(ticker, day)

    return image_paths


def backtest_capital_comparison_plot(ticker: str) -> str:
    """
    Create backtesting capital comparison plot.

    Args:
        ticker (str): ticker symbol.

    Returns:
        str: relative path of the generated image.
    """
    results_path = REPORTS_DIR / f"{ticker}_backtest_results.csv"

    if not results_path.exists():
        raise FileNotFoundError(f"Backtest results file not found: {results_path}")

    data = pd.read_csv(results_path)
    data["Date"] = pd.to_datetime(data["Date"])

    image_name = f"{ticker}_backtest_capital_comparison.svg"
    STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = STATIC_IMAGES_DIR / image_name

    plt.figure(figsize=(1200, 600, "px"))
    plt.plot(data["Date"], data["Strategy_Capital"], label="Strategy capital")
    plt.plot(data["Date"], data["Buy_Hold_Capital"], label="Buy and hold capital")
    plt.title(f"{ticker} Strategy vs Buy and Hold")
    plt.xlabel("Date")
    plt.ylabel("Capital")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    return f"images/{image_name}"


def backtest_market_return_plot(ticker: str) -> str:
    """
    Create market return violin plot grouped by signal.

    Args:
        ticker (str): stock ticker symbol.

    Returns:
        str: relative path of the generated image.
    """
    backtest_path = REPORTS_DIR / f"{ticker}_backtest_results.csv"

    if not backtest_path.exists():
        raise FileNotFoundError(f"Backtest results file not found: {backtest_path}")

    data = pd.read_csv(backtest_path)

    data["Signal"] = data["Signal"].map({
        1: "Invested",
        0: "Not invested"
    })

    image_name = f"{ticker}_backtest_market_return_violinplot.svg"
    STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = STATIC_IMAGES_DIR / image_name
    plt.figure(figsize=(1200, 600, "px"))
    sns.violinplot(data=data, x="Signal", y="Market_Return", inner="quartile")
    plt.title(f"{ticker} - Market Return Distribution")
    plt.xlabel("Capital condition")
    plt.ylabel("Next Day Market Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    return f"images/{image_name}"


def all_backtest_plots(ticker: str) -> dict:
    """
    Create all backtesting plots.

    Args:
        ticker (str): ticker symbol.

    Returns:
        dict: generated image paths.
    """
    image_paths = {
        "capital_comparison": backtest_capital_comparison_plot(ticker),
        "market_return": backtest_market_return_plot(ticker)
    }

    return image_paths
