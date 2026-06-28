# Stock Price Movement Prediction

This project is a small machine learning application for predicting stock price movements.

## Main Features

- Historical stock data download from Yahoo Finance.
- Data preprocessing with Pandas.
- Ridge Regression model training with scikit-learn.
- Prediction of the next 7 business days average prices.
- Visual evaluation plots.
- Simple trading strategy backtest.

## How to Run

### Run locally

```bash
pip install -r requirements.txt
python app/app.py
```

Then open the application at:

```text
http://localhost:5000
```

### Run with Docker

```bash
docker build -t stock-price-app .
docker run --rm -p 5000:5000 stock-price-app
```

Then open:

```text
http://localhost:5000
```

## How to Use the Application

1. Select a ticker symbol in the main page.
2. Click **Train** to download the latest data, process it, and train the model.
3. Click **Results** to see model evaluation plots and the next 7 days predictions.
4. Click **Backtest** to compare the model-based strategy with a buy-and-hold strategy.

## Backtesting

The backtest uses the predicted next-day average price to decide whether the capital should be invested or not. If the expected return is higher than a fixed threshold, the strategy invests; otherwise, it stays out of the market.