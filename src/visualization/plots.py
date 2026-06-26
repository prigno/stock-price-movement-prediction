import matplotlib
matplotlib.use("svg") # use a non interactive backend because want to save the files only

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

