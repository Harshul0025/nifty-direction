# Makes today's prediction for the website and saves it in site/data/live.json.
# GitHub Actions runs this every weekday:
#   about 7:50 AM IST  -> predict today (US has closed, Japan and Hong Kong have opened)
#   about 4:00 PM IST  -> write the real result next to each old prediction
#
# By hand:  python live/predict_today.py                     (same as above)
#           python live/predict_today.py --date 2026-11-13   (replay an old morning, marked "replay")
import json, os, sys
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from model import download_data, build_features, make_model, OVERNIGHT

LIVE_FILE = "site/data/live.json"
now = datetime.now(ZoneInfo("Asia/Kolkata"))
today = pd.Timestamp(now.date())

# ----- which day do we predict? -----
if "--date" in sys.argv:
    target, stage = pd.Timestamp(sys.argv[sys.argv.index("--date") + 1]), "replay"
elif now.weekday() < 5 and 7 <= now.hour and (now.hour, now.minute) < (9, 15):
    target, stage = today, "live"          # a weekday morning, before the market opens
else:
    target, stage = None, None             # not a prediction time: only fill in old results

raw = download_data()
nifty = raw["nifty"]
predictions = json.load(open(LIVE_FILE))["predictions"] if os.path.exists(LIVE_FILE) else []
done = [p["date"] for p in predictions]

# ----- 1. make the prediction -----
if target is not None and str(target.date()) not in done:
    # the model may only see days that finished BEFORE the target day
    raw["nifty"] = nifty[nifty.index < target]
    df, FEATURES = build_features(raw, next_day=target)
    data, predict_row = df.iloc[:-1].dropna(), df.iloc[[-1]]

    model = make_model().fit(data[FEATURES], data["target"])
    p_up = float(model.predict_proba(predict_row[FEATURES])[0, 1])

    predictions.append({
        "date": str(target.date()),
        "p_up": round(p_up, 3),
        "pred": "UP" if p_up > 0.5 else "DOWN",
        "confident": bool(p_up >= 0.60 or p_up <= 0.40),
        "stage": stage,                                              # "live" or "replay"
        "made_at": now.strftime("%Y-%m-%d %H:%M IST"),
        "trained_on_days": len(data),
        "last_finished_day": str(data.index.max().date()),
        "inputs": {name: round(float(predict_row[name].iloc[0]) * 100, 2) for name in OVERNIGHT},   # in percent
        "asia_open": {name: bool(target in raw[name].index) for name in ["nikkei", "hangseng"]},
        "actual": None, "ret": None, "correct": None,
    })
    print(f"Prediction for {target.date()}: P(UP) = {p_up:.3f}")

# ----- 2. fill in the real result of old predictions -----
closes = nifty["Close"]
for p in predictions:
    day = pd.Timestamp(p["date"])
    if p["actual"] is not None:
        continue
    finished = day < today or (now.hour, now.minute) >= (15, 45)       # NSE closes at 3:30 PM
    if day in closes.index and finished:
        ret = closes[day] / closes[closes.index < day].iloc[-1] - 1    # yesterday's close -> that day's close
        p["actual"] = "UP" if ret > 0 else "DOWN"
        p["ret"] = round(float(ret) * 100, 2)
        p["correct"] = bool(p["actual"] == p["pred"])
        print(f"Result for {p['date']}: {p['actual']} ({p['ret']:+.2f}%), prediction was {p['pred']}")
    elif day not in closes.index and (closes.index > day).any():
        p["actual"] = "CLOSED"                                         # the market was shut that day (holiday)

predictions.sort(key=lambda p: p["date"])
with open(LIVE_FILE, "w") as out:
    json.dump({"updated": now.strftime("%Y-%m-%d %H:%M IST"), "predictions": predictions}, out, indent=1)
