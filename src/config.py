from pathlib import Path


# project root
ROOT_DIR = Path(__file__).resolve().parents[1]

# available tickers
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]

# period considered
START_DATE = "2000-01-01"
END_DATE = "2026-06-01"

# previous days used as features
PREVIOUS_DAYS = list(range(1, 2))

# previous days used to compute statistics
WINDOW_SIZES = []

# statistics computed
STATISTICS = ["min", "max", "mean", "std"]

# feature columns used by model
FEATURE_COLUMNS = (
    ["avg_current"] +
    [f"avg_prev_{prev}" for prev in PREVIOUS_DAYS] +
    [f"stat_{statistic}_{window_size}" for window_size in WINDOW_SIZES for statistic in STATISTICS]
)

# number of days to predict
TARGET_DAYS = list(range(1, 8))
PREDICTION_DAYS = 7

# target columns to predict
TARGET_COLUMNS = [f"Target_Average_Price_{day}" for day in TARGET_DAYS]

# percentage of testing set
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.125

# ridge regularization parameter
ALPHA_VALUES = [i / 100 for i in range(1, 201)]

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

PREDICTIONS_DIR = ROOT_DIR / "data" / "predictions"

MODELS_DIR = ROOT_DIR / "src" / "models"

REPORTS_DIR = ROOT_DIR / "src" / "reports"

BACKTEST_DIR = ROOT_DIR / "src" / "backtest"

STATIC_IMAGES_DIR = ROOT_DIR / "app" / "static" / "images"