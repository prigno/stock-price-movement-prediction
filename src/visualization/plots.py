import matplotlib
matplotlib.use("svg") # use a non interactive backend because want to save the files only

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import STATIC_IMAGES_DIR, RAW_DATA_DIR, REPORTS_DIR, PREDICTIONS_DIR


def avg_price_plot(ticker: str) -> str:
    """
    Create average price plot.

    Args:
        ticker (str): ticker symbol.

    Returns:
        str: relative path of the generated image.
    """
    data_path = RAW_DATA_DIR / f"{ticker}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {data_path}")

    data = pd.read_csv(data_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")

    data["Average_Price"] = (data["High"] + data["Low"]) / 2

    image_name = f"{ticker}_average_price.svg"
    STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = STATIC_IMAGES_DIR / image_name

    plt.figure(figsize=(1200, 600, "px"))
    plt.plot(data["Date"], data["Average_Price"])
    plt.title(f"{ticker} Average Price")
    plt.xlabel("Date")
    plt.ylabel("Average Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    return f"images/{image_name}"


def model_evaluation_plot(ticker: str) -> str:
    """
    Create model evaluation plot.

    Args:
        ticker (str): ticker symbol.

    Returns:
        str: relative path of the generated image.
    """
    predictions_path = REPORTS_DIR / f"{ticker}_ridge_regression_test_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Test predictions file not found: {predictions_path}")

    data = pd.read_csv(predictions_path)
    data["Date"] = pd.to_datetime(data["Date"])

    image_name = f"{ticker}_model_evaluation.svg"
    STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = STATIC_IMAGES_DIR / image_name

    plt.figure(figsize=(1200, 600, "px"))
    plt.plot(data["Date"], data["Actual_Average_Price_1"], label="Real values")
    plt.plot(data["Date"], data["Predicted_Average_Price_1"], label="Predicted values")
    plt.title(f"{ticker} Model Evaluation")
    plt.xlabel("Date")
    plt.ylabel("Average Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    return f"images/{image_name}"


def error_plot(ticker: str) -> str:
    """
    Create absolute error plot.

    Args:
        ticker (str): stock ticker symbol.

    Returns:
        str: relative path of the generated image.
    """
    predictions_path = REPORTS_DIR / f"{ticker}_ridge_regression_test_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Test predictions file not found: {predictions_path}")

    data = pd.read_csv(predictions_path)
    data["Date"] = pd.to_datetime(data["Date"])

    image_name = f"{ticker}_absolute_error.svg"
    STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = STATIC_IMAGES_DIR / image_name

    plt.figure(figsize=(1200, 600, "px"))
    plt.plot(data["Date"], data["Absolute_Error"])
    plt.title(f"{ticker} Absolute Error")
    plt.xlabel("Date")
    plt.ylabel("Absolute Error")
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
        "average_price": avg_price_plot(ticker),
        "model_evaluation": model_evaluation_plot(ticker),
        "absolute_error": error_plot(ticker),
        "predictions": prediction_plot(ticker)
    }

    return image_paths

