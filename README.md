# NIFTY 50 Direction Prediction: a Pre-Market Model

**Harshul Saini & Vishesh Gupta**

Every morning at 9:00 AM, the model predicts whether NIFTY 50 will close today above yesterday's close (UP / DOWN).

The whole project is in one file: **`nifty_direction.ipynb`**. It contains the code, the charts, and the explanation of what was done, why, and why not something else.

**Live website: https://nifty-direction.vercel.app** (today's signal, the results, and a daily track record)

## Results (see the notebook for details)

| | Accuracy |
|---|---|
| Unseen test set (Sep 2024 to Nov 2026) | 58.6% (best simple rule 57.3%, always UP 49.1%) |
| Walk-forward 2019 to 2026 | 62.5% (best simple rule 59.6%) |
| Days when the model is confident | about 64% to 70% |

The notebook also shows honestly that this is a direction model, not a money-making strategy.

## How to run

```bash
cd ~/nifty-direction
source .venv/bin/activate
jupyter notebook nifty_direction.ipynb
```

Then choose **Run > Run All Cells**. It takes about 2 minutes.

- Prices are saved in `data/`, so the numbers are the same on every run.
- To use the latest prices, set `REFRESH_DATA = True` in Section 0. The written findings quote the numbers of the saved data, so they will differ slightly after a refresh.

### Windows Setup (PowerShell)

From the repository root, create a virtual environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install nbconvert ipykernel
```

Keep `REFRESH_DATA = False` in Section 0 to use the saved prices. Run all cells from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600 --output verified_run.ipynb --output-dir .venv .\nifty_direction.ipynb
```

The executed copy is saved to `.venv\verified_run.ipynb`.

## Where each required component is

| Component | Notebook section |
|---|---|
| Exploratory Data Analysis | 1 |
| Feature Derivation | 2 |
| Classification | 3 |
| Regularization | 4 |
| Hyperparameter Tuning | 5 |
| Ensemble Techniques | 6 |
| Performance Analysis | 7 |
| Real-life checks and the next signal | 8 |
| Conclusion and viva questions | 9 |

## The live website

The website shows the same model and the same numbers as the notebook.

- `site/` is a plain HTML, CSS and JavaScript page. Vercel serves this folder.
- `live/model.py` builds the features and the final voting model, exactly as in the notebook.
- `live/make_results.py` repeats the test and the walk-forward check on the saved data and writes `site/data/results.json` (the charts).
- `live/predict_today.py` downloads fresh prices, trains the model on every finished day and writes today's prediction to `site/data/live.json`.
- `.github/workflows/daily.yml` runs `predict_today.py` every weekday at about 7:50 AM IST (prediction) and 4:00 PM IST (real result).

A prediction is saved before the market opens and is never changed later. Rows marked "replay" were made after the day, using only the data that was available that morning.

To see the site on your own computer:

```bash
cd site
python3 -m http.server 8000      # then open http://localhost:8000
```

## Files

- `nifty_direction.ipynb`: the project.
- `data/`: saved prices from Yahoo Finance (10 series).
- `requirements.txt`: Python packages.
- `site/`, `live/`, `.github/`, `vercel.json`: the live website (see above).
- `old/`: the first version of the project (prediction at 3:30 PM the day before, about 52% accuracy), kept for reference.
