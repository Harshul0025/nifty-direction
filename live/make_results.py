# Makes site/data/results.json: the numbers and chart data shown on the website.
# It repeats the notebook's test (Section 7) and walk-forward check (Section 8) on the saved data,
# so the website shows the same numbers as the notebook and the report.
# Run from the project folder:  python live/make_results.py
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
from model import load_saved_data, build_features, make_model, OVERNIGHT, SEED

df, FEATURES = build_features(load_saved_data())
data = df.iloc[:-1].dropna()
split = int(len(data) * 0.8)
train, test = data.iloc[:split], data.iloc[split:]
X_train, y_train, X_test, y_test = train[FEATURES], train["target"], test[FEATURES], test["target"]


def pct(x):
    return round(float(x) * 100, 1)


def baselines(d):
    return {
        "Rule: always UP": np.ones(len(d), dtype=int),
        "Rule: same as yesterday": (d["prev_ret_1"] > 0).astype(int).values,
        "Rule: follow the US": (d["sp500_night"] > 0).astype(int).values,
    }


# ----- every model on the test set -----
voting = make_model()
models = {
    "Logistic Regression": clone(voting.estimators[0][1]),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=SEED),
    "Random Forest": clone(voting.estimators[1][1]),
    "XGBoost": clone(voting.estimators[2][1]),
    "Voting (final model)": voting,
}
tscv = TimeSeriesSplit(n_splits=5)
rows = []
for name, model in models.items():
    val_auc = cross_validate(model, X_train, y_train, cv=tscv, scoring="roc_auc")["test_score"].mean()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    rows.append({"name": name, "kind": "final" if model is voting else "model",
                 "test_acc": pct(accuracy_score(y_test, proba > 0.5)),
                 "test_auc": round(float(roc_auc_score(y_test, proba)), 3),
                 "val_auc": round(float(val_auc), 3)})
for name, pred in baselines(test).items():
    rows.append({"name": name, "kind": "rule", "test_acc": pct(accuracy_score(y_test, pred))})

# ----- final model: confusion matrix, confidence, importance, ablation -----
test_proba = voting.predict_proba(X_test)[:, 1]
test_pred = (test_proba > 0.5).astype(int)
confusion = {
    "up_right": int(((test_pred == 1) & (y_test == 1)).sum()), "up_wrong": int(((test_pred == 1) & (y_test == 0)).sum()),
    "down_right": int(((test_pred == 0) & (y_test == 0)).sum()), "down_wrong": int(((test_pred == 0) & (y_test == 1)).sum()),
}

confidence = []
for level in [0.00, 0.05, 0.10, 0.15, 0.20]:
    sure = np.abs(test_proba - 0.5) >= level
    confidence.append({"label": "All days" if level == 0 else f"P(UP) above {0.5 + level:.2f} or below {0.5 - level:.2f}",
                       "days": int(sure.sum()), "share": round(float(sure.mean()) * 100),
                       "acc": pct(accuracy_score(y_test[sure], test_pred[sure]))})

imp = permutation_importance(voting, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED)
importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).head(8)
importance = [{"name": k, "drop": round(float(v), 3), "overnight": k in OVERNIGHT} for k, v in importance.items()]

YESTERDAY_ONLY = [c for c in FEATURES if c not in OVERNIGHT]
old_proba = make_model().fit(X_train[YESTERDAY_ONLY], y_train).predict_proba(X_test[YESTERDAY_ONLY])[:, 1]

# ----- walk-forward: re-train every 63 trading days, always using only the past -----
X_all, y_all = data[FEATURES], data["target"]
start = data.index.searchsorted(pd.Timestamp("2019-01-01"))
wf_proba = []
for s in range(start, len(data), 63):
    e = min(s + 63, len(data))
    model = make_model().fit(X_all.iloc[:s], y_all.iloc[:s])
    wf_proba.extend(model.predict_proba(X_all.iloc[s:e])[:, 1])
wf = data.iloc[start:].copy()
wf["proba"] = wf_proba
wf["pred"] = (wf["proba"] > 0.5).astype(int)
wf["hit"] = wf["pred"] == wf["target"]
wf_sure = np.abs(wf["proba"] - 0.5) >= 0.10

by_year = pd.DataFrame({"model": wf["hit"], "follow_us": baselines(wf)["Rule: follow the US"] == wf["target"]})
yearly = by_year.groupby(by_year.index.year).mean()

# ----- money check: growth of 1 rupee -----
says_up = wf["pred"] == 1
COST = 0.0003 * 2
curves = pd.DataFrame({
    "impossible": (1 + np.where(says_up, wf["today_ret"], 0)).cumprod(),
    "real": (1 + np.where(says_up, wf["open_to_close"] - COST, 0)).cumprod(),
    "hold": (1 + wf["today_ret"]).cumprod(),
}, index=wf.index)
monthly = curves.resample("ME").last()
monthly.loc[curves.index[-1]] = curves.iloc[-1]          # make the last point the true last day
monthly = monthly[~monthly.index.duplicated(keep="last")].sort_index()

results = {
    "data": {"days": len(data), "first_day": str(data.index.min().date()), "last_day": str(data.index.max().date()),
             "train_days": len(train), "test_days": len(test),
             "test_from": str(test.index.min().date()), "test_to": str(test.index.max().date()),
             "test_up_share": pct(y_test.mean()), "features": len(FEATURES)},
    "models": rows,
    "confusion": confusion,
    "confidence": confidence,
    "importance": importance,
    "ablation": {"yesterday_only_acc": pct(accuracy_score(y_test, old_proba > 0.5)),
                 "yesterday_only_auc": round(float(roc_auc_score(y_test, old_proba)), 3),
                 "full_acc": pct(accuracy_score(y_test, test_pred)),
                 "full_auc": round(float(roc_auc_score(y_test, test_proba)), 3)},
    "walk_forward": {
        "from": str(wf.index.min().date()), "to": str(wf.index.max().date()), "days": len(wf),
        "acc": pct(wf["hit"].mean()), "auc": round(float(roc_auc_score(wf["target"], wf["proba"])), 3),
        "follow_us_acc": pct(by_year["follow_us"].mean()),
        "confident_share": round(float(wf_sure.mean()) * 100), "confident_acc": pct(wf["hit"][wf_sure].mean()),
        "years": [int(y) for y in yearly.index],
        "model_by_year": [pct(v) for v in yearly["model"]],
        "follow_us_by_year": [pct(v) for v in yearly["follow_us"]],
    },
    "money": {
        "open_to_close_acc": pct(((wf["open_to_close"] > 0) == says_up).mean()),
        "final": {k: round(float(v), 2) for k, v in curves.iloc[-1].items()},
        "dates": [str(d.date()) for d in monthly.index],
        "impossible": [round(float(v), 3) for v in monthly["impossible"]],
        "real": [round(float(v), 3) for v in monthly["real"]],
        "hold": [round(float(v), 3) for v in monthly["hold"]],
    },
    # last 60 days of the walk-forward check, for the hit / miss strip
    "recent": [{"date": str(d.date()), "p_up": round(float(r["proba"]), 2), "hit": bool(r["hit"]),
                "ret": round(float(r["today_ret"]) * 100, 2)} for d, r in wf.tail(60).iterrows()],
}

with open("site/data/results.json", "w") as out:
    json.dump(results, out, indent=1)

print("Test accuracy:", results["ablation"]["full_acc"], "| AUC:", results["ablation"]["full_auc"])
print("Walk-forward:", results["walk_forward"]["acc"], "| confident days:", results["walk_forward"]["confident_acc"])
print("1 rupee becomes:", results["money"]["final"])
for r in rows:
    print(r)
