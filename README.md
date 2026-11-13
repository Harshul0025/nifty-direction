# NIFTY 50 Direction Prediction: a Pre-Market Model

**Harshul Saini & Vishesh Gupta**

Every morning at 9:00 AM, the model predicts whether NIFTY 50 will close today above yesterday's close (UP / DOWN).

The whole project is in one file: **`nifty_direction.ipynb`**. It contains the code, the charts, and the explanation of what was done, why, and why not something else.

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

## Files

- `nifty_direction.ipynb`: the project.
- `data/`: saved prices from Yahoo Finance (10 series).
- `requirements.txt`: Python packages.
- `old/`: the first version of the project (prediction at 3:30 PM the day before, about 52% accuracy), kept for reference.
