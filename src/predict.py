# src/predict.py
import pandas as pd
import joblib
import glob
import pathlib
import math
from datetime import datetime
from sklearn.impute import SimpleImputer
from src.file_manager import save_pick_json, list_today_picks, load_pick
from src.math_core import poisson_pmf, kelly_fraction, expected_value
from src.history import update_weight_adjustments

ROOT = pathlib.Path(".")
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

def load_latest_model():
    files = sorted(MODELS_DIR.glob("top5_leagues_model_*.pkl"))
    if not files:
        return None
    return joblib.load(files[-1])

def load_latest_odds():
    files = sorted(DATA_DIR.glob("odds_*.csv"))
    if not files:
        return None
    return pd.read_csv(files[-1])

def find_existing_pick_for_match(match_id):
    for p in list_today_picks():
        try:
            obj = load_pick(p)
            if str(obj.get("match_id")) == str(match_id):
                return obj, p
        except Exception:
            continue
    return None, None

def significant_ev_improvement(old_ev, new_ev, threshold=0.05):
    """
    Regla: re-emisión permitida si EV mejora en al menos threshold (absolute).
    """
    if old_ev is None:
        return True
    try:
        return (new_ev - old_ev) >= threshold
    except Exception:
        return False

def predict_and_save(window_label="auto"):
    """
    Ejecuta predicciones y guarda picks en picks_diarios con deduplicación y regla anti-spam.
    window_label: '08', '12', '16' or 'auto'
    """
    model = load_latest_model()
    odds_df = load_latest_odds()
    if odds_df is None or odds_df.empty:
        print("No hay datos de odds disponibles.")
        return []

    # Ajustes desde historial
    weights = update_weight_adjustments()
    poisson_weight = weights.get("poisson_weight", 1.0)

    # Features esperadas (si el modelo existe)
    features = ["FTHG","FTAG","HS","AS","HST","AST","HC","AC","HF","AF","HY","AY","HR","AR"]
    if "xG" in odds_df.columns:
        features.append("xG")
    if "xGA" in odds_df.columns:
        features.append("xGA")
    if "xG" in odds_df.columns and "xGA" in odds_df.columns:
        odds_df["xG_diff"] = odds_df["xG"] - odds_df["xGA"]
        features.append("xG_diff")

    available = [c for c in features if c in odds_df.columns]

    results = []
    # If model exists and features available, use it; else fallback to implied probs from odds
    if model is not None and len(available) > 0:
        X = odds_df[available]
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=available)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        for idx, row in odds_df.iterrows():
            match_id = row.get("match_id", f"{row.get('HomeTeam','?')}_vs_{row.get('AwayTeam','?')}_{idx}")
            home = row.get("HomeTeam", "Home")
            away = row.get("AwayTeam", "Away")
            prob_local = float(probs[idx][1])
            # Example: use odds column if present
            odds = row.get("home_odds") or row.get("h2h_home") or row.get("odds_home") or 2.0
            ev = expected_value(prob_local, float(odds))
            kelly = kelly_fraction(prob_local, float(odds), fraction=0.25)
            # Poisson lambda adjusted by weight
            poisson_lambda = max(0.01, prob_local * 2.0 * poisson_weight)
            pick = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "window": window_label,
                "match_id": match_id,
                "home_team": home,
                "away_team": away,
                "pick": "Local gana" if preds[idx] == 1 else "Visita gana",
                "prob_pick": prob_local,
                "odds": float(odds),
                "ev": ev,
                "kelly_fraction": kelly,
                "poisson_lambda": poisson_lambda
            }
            # Deduplication & anti-spam
            existing, path = find_existing_pick_for_match(match_id)
            if existing:
                old_ev = existing.get("ev")
                if significant_ev_improvement(old_ev, ev):
                    save_pick_json(pick)
                    results.append(pick)
                else:
                    # skip re-send
                    continue
            else:
                save_pick_json(pick)
                results.append(pick)
    else:
        # Fallback: compute implied probabilities from odds
        for idx, row in odds_df.iterrows():
            match_id = row.get("match_id", f"{row.get('HomeTeam','?')}_vs_{row.get('AwayTeam','?')}_{idx}")
            home = row.get("HomeTeam", "Home")
            away = row.get("AwayTeam", "Away")
            # Try to extract h2h odds from common columns
            odd_home = row.get("home_odds") or row.get("h2h_home") or row.get("odds_home")
            odd_away = row.get("away_odds") or row.get("h2h_away") or row.get("odds_away")
            if pd.isna(odd_home) or pd.isna(odd_away) or odd_home is None or odd_away is None:
                continue
            try:
                p_home = 1.0 / float(odd_home)
                p_away = 1.0 / float(odd_away)
            except Exception:
                continue
            # Normalize
            s = p_home + p_away
            if s <= 0:
                continue
            p_home /= s
            p_away /= s
            pick_choice = "Local gana" if p_home >= p_away else "Visita gana"
            odds_used = float(odd_home) if pick_choice == "Local gana" else float(odd_away)
            prob_pick = p_home if pick_choice == "Local gana" else p_away
            ev = expected_value(prob_pick, odds_used)
            kelly = kelly_fraction(prob_pick, odds_used, fraction=0.25)
            poisson_lambda = max(0.01, prob_pick * 2.0 * poisson_weight)
            pick = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "window": window_label,
                "match_id": match_id,
                "home_team": home,
                "away_team": away,
                "pick": pick_choice,
                "prob_pick": prob_pick,
                "odds": odds_used,
                "ev": ev,
                "kelly_fraction": kelly,
                "poisson_lambda": poisson_lambda
            }
            existing, path = find_existing_pick_for_match(match_id)
            if existing:
                old_ev = existing.get("ev")
                if significant_ev_improvement(old_ev, ev):
                    save_pick_json(pick)
                    results.append(pick)
                else:
                    continue
            else:
                save_pick_json(pick)
                results.append(pick)
    return results

if __name__ == "__main__":
    import json
    res = predict_and_save(window_label="auto")
    print(f"Generated {len(res)} picks")
