# Shared code for the live website.
# Same data, same features and same final model as the notebook (nifty_direction.ipynb).
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

SEED = 42
TICKERS = {
    "nifty": "^NSEI", "indiavix": "^INDIAVIX", "banknifty": "^NSEBANK",        # India
    "sp500": "^GSPC", "nasdaq": "^IXIC", "usvix": "^VIX",                      # USA
    "crude": "CL=F", "usdinr": "INR=X",                                        # oil and rupee
    "nikkei": "^N225", "hangseng": "^HSI",                                     # Asia
}
OVERNIGHT = ["sp500_night", "usvix_night", "crude_night", "usdinr_night", "nikkei_gap", "hangseng_gap"]
DROPPED = ["prev_range_14", "nasdaq_night"]      # near-duplicate features, removed in notebook Section 2i


def load_saved_data(folder="data"):
    # the CSV files that the notebook saved
    return {name: pd.read_csv(f"{folder}/{name}.csv", index_col=0, parse_dates=True) for name in TICKERS}


def download_data():
    # fresh prices from Yahoo Finance (tries 3 times, because Yahoo sometimes refuses)
    raw = {}
    for name, ticker in TICKERS.items():
        for attempt in range(3):
            d = yf.download(ticker, start="2015-01-01", auto_adjust=True, progress=False)
            if len(d) > 0:
                break
            time.sleep(20)
        if len(d) == 0:
            raise RuntimeError(f"Could not download {name} ({ticker})")
        d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None)
        raw[name] = d.dropna(subset=["Close"])
    return raw


def build_features(raw, next_day=None):
    # Returns (df, FEATURES). The last row of df is the day to predict: features ready, answer unknown.
    nifty = raw["nifty"].copy()
    nifty["Volume"] = nifty["Volume"].replace(0, np.nan).ffill().bfill()
    if next_day is None:
        next_day = nifty.index[-1] + pd.offsets.BDay(1)
    nifty.loc[next_day] = np.nan

    close, open_, high, low, volume = nifty["Close"], nifty["Open"], nifty["High"], nifty["Low"], nifty["Volume"]

    # NIFTY features, calculated at the END of each day
    f = pd.DataFrame(index=nifty.index)
    f["ret_1"] = close.pct_change()
    f["ret_5"] = close.pct_change(5)
    f["ret_20"] = close.pct_change(20)
    f["vs_sma20"] = close / close.rolling(20).mean() - 1
    f["vs_sma50"] = close / close.rolling(50).mean() - 1
    f["vs_sma200"] = close / close.rolling(200).mean() - 1

    change = close.diff()
    avg_gain = change.clip(lower=0).rolling(14).mean()
    avg_loss = (-change.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + avg_gain / avg_loss)

    macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / close

    f["volatility_20"] = f["ret_1"].rolling(20).std()
    f["range_14"] = ((high - low) / close).rolling(14).mean()
    f["gap"] = open_ / close.shift(1) - 1
    f["intraday_ret"] = close / open_ - 1
    f["close_position"] = (close - low) / (high - low)
    f["volume_ratio"] = volume / volume.rolling(20).mean()
    f["dist_52w_high"] = close / close.rolling(252).max() - 1

    india_vix = raw["indiavix"]["Close"].reindex(nifty.index).ffill()
    f["india_vix"] = india_vix
    f["india_vix_change"] = india_vix.pct_change()
    f["banknifty_ret"] = raw["banknifty"]["Close"].reindex(nifty.index).ffill().pct_change()

    # SHIFT BY ONE DAY: today's row gets yesterday's values (the leakage guard)
    df = f.shift(1).add_prefix("prev_")
    df["day_of_week"] = df.index.dayofweek

    # Overnight features: a US closing price of date D is known in India only on the morning of D+1
    def known_before_9am(series):
        s = series.copy()
        s.index = s.index + pd.Timedelta(days=1)
        return s.reindex(nifty.index, method="ffill")

    for name in ["sp500", "nasdaq", "usvix", "crude", "usdinr"]:
        df[name + "_night"] = known_before_9am(raw[name]["Close"]).pct_change()
    df["crude_night"] = df["crude_night"].clip(-0.2, 0.2)

    # Asia this morning: opening gap of Japan and Hong Kong (0 if that market is closed today)
    for name in ["nikkei", "hangseng"]:
        asia = raw[name]
        asia_gap = asia["Open"] / asia["Close"].shift(1) - 1
        df[name + "_gap"] = asia_gap.reindex(df.index).fillna(0)

    features = [c for c in df.columns if c not in DROPPED]

    # The answer (target). These three columns are NOT features.
    df["today_ret"] = close.pct_change()
    df["target"] = (df["today_ret"] > 0).astype(int)
    df["open_to_close"] = close / open_ - 1
    return df, features


def make_model():
    # The final model of the notebook: soft voting of the three tuned models (settings from Section 5)
    log_reg = make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.01, l1_ratio=0, solver="liblinear", max_iter=1000))
    forest = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=SEED, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.01,
                        subsample=0.8, colsample_bytree=0.8, random_state=SEED)
    return VotingClassifier(estimators=[("lr", log_reg), ("rf", forest), ("xgb", xgb)], voting="soft")
