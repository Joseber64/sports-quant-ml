# src/file_manager.py
import pathlib
import shutil
import json
from datetime import datetime

ROOT = pathlib.Path(".")
PICKS_DIR = ROOT / "picks_diarios"
HIST_DIR = ROOT / "archivo_historico"

PICKS_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)

def save_pick_json(pick: dict, timestamped: bool = True):
    """
    Guarda un pick en picks_diarios como JSON.
    pick debe contener al menos: match_id, home_team, away_team, pick, prob_pick, odds, kelly_fraction
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S") if timestamped else ""
    filename = f"pick_{pick.get('match_id','unknown')}_{ts}.json" if ts else f"pick_{pick.get('match_id','unknown')}.json"
    path = PICKS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pick, f, ensure_ascii=False, indent=2)
    return path

def list_today_picks():
    """
    Lista picks en picks_diarios creados hoy (UTC).
    """
    files = sorted(PICKS_DIR.glob("pick_*.json"))
    return files

def load_pick(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def move_to_historic(path, date_str=None):
    """
    Mueve un archivo de picks_diarios a archivo_historico renombrándolo a picks_YYYY_MM_DD.json (agrupa por fecha).
    Si date_str no se pasa, usa fecha UTC actual.
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y_%m_%d")
    dest_name = f"picks_{date_str}.json"
    dest_path = HIST_DIR / dest_name
    # Si existe, append as array; else create array
    import json
    if dest_path.exists():
        with open(dest_path, "r", encoding="utf-8") as f:
            try:
                arr = json.load(f)
                if not isinstance(arr, list):
                    arr = [arr]
            except Exception:
                arr = []
    else:
        arr = []
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    arr.append(obj)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)
    # Remove original
    path.unlink()
    return dest_path

def clear_picks_dir():
    """
    Borra todos los archivos en picks_diarios (usado al final del día).
    """
    for p in PICKS_DIR.glob("pick_*.json"):
        try:
            p.unlink()
        except Exception:
            pass
