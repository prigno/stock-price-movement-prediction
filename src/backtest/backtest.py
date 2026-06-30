from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.config import REPORTS_DIR, TICKERS


INITIAL_CAPITAL = 10000
THRESHOLD = 0.001   # buy only if tomorrow's avg price is at least 0.1% higher that today's avg price
TRANSACTION_COST = 0.001    # everytime a bought or a sold is done, a transaction of 0.1% is paid


def _load_test_predictions(ticker: str) -> pd.DataFrame:
    """
    Load test predictions of a ticker.

    Args:
        ticker (str): ticker symbol.

    Returns:
        pd.DataFrame: test predictions.
    """
    file_path = REPORTS_DIR / f"{ticker}_ridge_regression_test_predictions.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Test predictions file not found: {file_path}")

    # columns needed for backtest
    columns = ["Date", "Current_Average_Price", "Actual_Average_Price_1", "Predicted_Average_Price_1"]

    return pd.read_csv(file_path, usecols=columns)


def _add_trading_signals(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Add trading signals to data.

    Args:
        data (pd.DataFrame): test predictions.
        threshold (float): minimum expected return required to buy.

    Returns:
        pd.DataFrame: data with trading signals.
    """
    data = data.copy()

    # expected return = (tomorrow's predicted avg price - today's actual avg price) / today's actual avg price
    data["Expected_Return"] = (data["Predicted_Average_Price_1"] - data["Current_Average_Price"]) / data["Current_Average_Price"]
    # Signal = 1 if Expected Return > Threshold, 0 otherwise
    data["Signal"] = (data["Expected_Return"] > threshold).astype(int)

    return data


def _calculate_returns(data: pd.DataFrame, initial_capital: float, transaction_cost: float) -> pd.DataFrame:
    """
    Calculate strategy and buy-and-hold returns.

    Args:
        data (pd.DataFrame): data with trading signals.
        initial_capital (float): initial available capital.
        transaction_cost (float): cost applied when buy or sell.

    Returns:
        pd.DataFrame: data with backtesting results.
    """
    data = data.copy()

    # Marker Return = (tomorrow's actual avg price - today's actual avg price) / today's actual avg price
    data["Market_Return"] = (data["Actual_Average_Price_1"] - data["Current_Average_Price"]) / data["Current_Average_Price"]

    # calculate the absolute differenze between signal in row x and signal in row x - 1
    # if trade = 0 -> stay
    # if trade = 1 -> buy or sell 
    trade = data["Signal"].diff().abs()
    # first raw is NaN (because don't have a previous raw) -> is filled with value of signal (1=buy, 0=don't buy)
    trade = trade.fillna(data["Signal"])

    # signal = 1 -> capital invested -> take the return
    # signal = 0 -> capital non invested -> don't take the return
    data["Strategy_Return"] = data["Signal"] * data["Market_Return"]
    data["Strategy_Return"] = data["Strategy_Return"] - trade * transaction_cost

    # final capital: initial capital * (1 + cumulative product of returns)
    data["Strategy_Capital"] = initial_capital * (1 + data["Strategy_Return"]).cumprod()
    data["Buy_Hold_Capital"] = initial_capital * (1 + data["Market_Return"]).cumprod()

    return data


def _calculate_metrics(data: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """
    Calculate backtesting metrics.

    Args:
        data (pd.DataFrame): backtesting results.
        initial_capital (float): initial available capital.

    Returns:
        pd.DataFrame: backtesting metrics.
    """
    final_strategy_capital = data["Strategy_Capital"].iloc[-1]
    final_buy_hold_capital = data["Buy_Hold_Capital"].iloc[-1]

    strategy_return = (final_strategy_capital - initial_capital) / initial_capital
    buy_hold_return = (final_buy_hold_capital - initial_capital) / initial_capital

    # count the number of boughts and solds
    trade = data["Signal"].diff().abs()
    trade = trade.fillna(data["Signal"])
    number_of_trades = trade.sum()

    # count the number of days in which the capital was invested
    invested_days = len(data[data["Signal"] == 1])

    # max capital reached for each day (cumulative max)
    strategy_peak = data["Strategy_Capital"].cummax()
    # calculate how much the capital has decreased realtive to the previous peak
    drawdown = (data["Strategy_Capital"] - strategy_peak) / strategy_peak
    # calculate the worst drawdown
    max_drawdown = drawdown.min()

    metrics = pd.DataFrame({
        "metric": [
            "initial_capital",
            "final_strategy_capital",
            "final_buy_hold_capital",
            "strategy_return",
            "buy_hold_return",
            "number_of_trades",
            "invested_days",
            "max_drawdown"
        ],
        "value": [
            initial_capital,
            final_strategy_capital,
            final_buy_hold_capital,
            strategy_return,
            buy_hold_return,
            number_of_trades,
            invested_days,
            max_drawdown
        ]
    })

    return metrics


def _save_backtest_results(ticker: str, data: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """
    Save backtesting results and metrics.

    Args:
        ticker (str): ticker symbol.
        data (pd.DataFrame): backtesting results.
        metrics (pd.DataFrame): backtesting metrics.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = REPORTS_DIR / f"{ticker}_backtest_results.csv"
    metrics_path = REPORTS_DIR / f"{ticker}_backtest_metrics.csv"

    data.to_csv(results_path, index=False)
    metrics.to_csv(metrics_path, index=False)


def run_backtest(ticker: str, threshold: float = THRESHOLD, initial_capital: float = INITIAL_CAPITAL, transaction_cost: float = TRANSACTION_COST) -> pd.DataFrame:
    """
    Run backtesting for a ticker.

    Args:
        ticker (str): ticker symbol.
        threshold (float): minimum expected return required to buy.
        initial_capital (float): initial available capital.
        transaction_cost (float): cost applied when buy or sell.

    Returns:
        pd.DataFrame: backtesting metrics.
    """
    data = _load_test_predictions(ticker)
    data = _add_trading_signals(data, threshold)
    data = _calculate_returns(data, initial_capital, transaction_cost)

    metrics = _calculate_metrics(data, initial_capital)

    _save_backtest_results(ticker, data, metrics)

    return metrics


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

    run_backtest(ticker)

