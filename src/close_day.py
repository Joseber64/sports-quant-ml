# src/close_day.py
import json
from datetime import datetime
from src.file_manager import list_today_picks, load_pick, move_to_historic, clear_picks_dir
from src.history import append_record, compute_brier
import pathlib

def fetch_real_result_for_match(match_id, home_team=None, away_team=None):
    """
    Placeholder: implement real result lookup (API or local dataset).
    For now, returns None (unknown). The user should replace this with real source.
    """
    # Example return: {"result": 1} where 1 means home win, 0 away
    return None

def close_day_process():
    files = list_today_picks()
    if not files:
        print("No picks to close.")
        return
    date_str = datetime.utcnow().strftime("%Y_%m_%d")
    for p in files:
        try:
            pick = load_pick(p)
            match_id = pick.get("match_id")
            home = pick.get("home_team")
            away = pick.get("away_team")
            prob = pick.get("prob_pick")
            odds = pick.get("odds")
            # Fetch real result
            real = fetch_real_result_for_match(match_id, home, away)
            if real is None:
                # If result unknown, mark as pending and still move to historic
                result = None
                brier = None
            else:
                result = real.get("result")
                brier = compute_brier(result, prob)
            record = {
                "date": datetime.utcnow().isoformat(),
                "match_id": match_id,
                "home_team": home,
                "away_team": away,
                "pick": pick.get("pick"),
                "prob_pick": prob,
                "odds": odds,
                "result": result,
                "brier_error": brier,
                "kelly_fraction": pick.get("kelly_fraction"),
                "poisson_lambda": pick.get("poisson_lambda"),
                "weight_adjustment": None
            }
            append_record(record)
            # Move file to historic (grouped by date)
            move_to_historic(p, date_str=date_str)
        except Exception as e:
            print("Error processing pick:", p, e)
    # After processing all, ensure picks_diarios is empty
    clear_picks_dir()
    print("Close day process completed.")

if __name__ == "__main__":
    close_day_process()
