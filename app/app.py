from datetime import date, timedelta    # date -> to obtain current date
                                        # timedelta -> to do subtraction with dates
from pathlib import Path
import sys

from flask import Flask, flash, redirect, render_template, request, url_for

 
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .resolve() -> abosolute path of current python file
sys.path.append(str(PROJECT_ROOT)) # add the project root to the paths in which modules are searched

from src.data.data_loader import load_and_save_ticker
from src.data.data_processor import process_and_save_ticker
from src.models.ridge_regression import train_and_evaluate_model

from src.models.ridge_prediction import predict_next_7_days
from src.visualization.plots import all_plots


app = Flask(__name__)
# !-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!#
app.secret_key = "stock-price-secret-key"
# !-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!!-!-!#


# available tickers
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]

# first date of the period considered
START_DATE = "2000-01-01"


# link the URL / to the function inedx()
@app.route("/", methods=["GET"])
def index():
    """
    Show the main page of the web application.
    """
    # read the ticker parameter from the URL. "" is the default if it's not specified
    selected_ticker = request.args.get("ticker", "")

    # return app/templates/index.html
    return render_template("index.html", tickers=TICKERS, selected_ticker=selected_ticker)

# link the URL /train to the function train()
@app.route("/train", methods=["POST"])
def train():
    """
    Train the model for the selected ticker.
    """
    # read the ticker specified in the HTML form
    ticker = request.form.get("ticker")

    if ticker not in TICKERS:
        flash("Please select a valid ticker before training.")
        return redirect(url_for("index"))

    try:
        # yfinance does not include the end date, so tomorrow is used
        # to include all available data up to today
        end_date = (date.today() + timedelta(days=1)).isoformat()

        load_and_save_ticker(ticker, start_date=START_DATE, end_date=end_date)
        process_and_save_ticker(ticker)
        train_and_evaluate_model(ticker)

        flash(f"Training completed for {ticker}.")

    except FileNotFoundError as error:
        flash(str(error))

    return redirect(url_for("index", ticker=ticker))


# link the URL /results to the function results()
@app.route("/results", methods=["GET"])
def results():
    """
    Show the results page for the selected ticker.
    """
    ticker = request.args.get("ticker")

    if ticker not in TICKERS:
        flash("Please select a valid ticker before showing results.")
        return redirect(url_for("index"))

    try:
        predict_next_7_days(ticker)
        image_paths = all_plots(ticker)

    except FileNotFoundError as error:
        flash(str(error))
        return redirect(url_for("index", ticker=ticker))

    return render_template("results.html", ticker=ticker, image_paths=image_paths)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

