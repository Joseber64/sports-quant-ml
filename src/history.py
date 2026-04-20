# src/history.py
import pandas as pd
import pathlib
from datetime import datetime

ROOT = pathlib.Path(".")
HISTORY_CSV = ROOT / "archivo_historico" / "history_master.csv"
pathlib.Path(ROOT / "archivo_historico").mkdir(parents=True, exist_ok=True)

HISTORY_COLUMNS = [
    "date", "match_id", "home_team", "away_team",
    "pick", "prob_pick", "odds", "result", "brier_error",
    "kelly_fraction", "poisson_lambda", "weight_adjustment"
]

def _ensure_history():
    if not HISTORY_CSV.exists():
        df = pd.DataFrame(columns=HISTORY_COLUMNS)
        df.to_csv(HISTORY_CSV, index=False)

def append_record(record: dict):
    """
    record keys must match HISTORY_COLUMNS (missing keys will be filled with None)
    """
    _ensure_history()
    df = pd.read_csv(HISTORY_CSV)
    row = {k: record.get(k, None) for k in HISTORY_COLUMNS}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)

def load_history():
    _ensure_history()
    return pd.read_csv(HISTORY_CSV)

def compute_brier(true, prob):
    """
    Brier score for binary outcome: (forecast - outcome)^2
    true: 0 or 1
    prob: predicted probability for the event
    """
    return (prob - float(true)) ** 2

def update_weight_adjustments():
    """
    Example: compute simple per-team or global weight adjustments based on historical Brier.
    This function returns a dict with global 'poisson_weight' to be used in predictions.
    """
    df = load_history()
    if df.empty:
        return {"poisson_weight": 1.0}
    # Use mean brier to scale poisson weight (lower brier -> trust model more)
    mean_brier = df["brier_error"].dropna().mean() if "brier_error" in df.columns else None
    if pd.isna(mean_brier) or mean_brier is None:
        return {"poisson_weight": 1.0}
    # Map mean_brier in [0,1] to weight in [0.5,1.5] (example heuristic)
    w = max(0.5, min(1.5, 1.0 + (0.5 - mean_brier)))
    return {"poisson_weight": w}
