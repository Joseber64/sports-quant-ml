# src/send_telegram.py
import os
import requests
import glob
import pandas as pd
from datetime import datetime
from src.file_manager import list_today_picks, load_pick

def send_telegram_message(message: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=data, timeout=15)
        print("Telegram status:", r.status_code)
        return r.status_code == 200
    except Exception as e:
        print("Telegram error:", e)
        return False

def build_message_from_pick(pick):
    home = pick.get("home_team")
    away = pick.get("away_team")
    pred = pick.get("pick")
    prob = pick.get("prob_pick")
    kelly = pick.get("kelly_fraction")
    odds = pick.get("odds")
    msg = f"*{home}* vs *{away}* — {pred}\nProb: `{prob:.2f}`  •  Odds: `{odds}`  •  Kelly: `{kelly:.2f}`\n"
    return msg

def send_daily_picks():
    files = list_today_picks()
    if not files:
        print("No hay predicciones para enviar")
        return False
    # Build a single message with all new picks that haven't been sent (dedup handled earlier)
    message = f"*📊 Picks del día — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*\n\n"
    for p in files:
        try:
            pick = load_pick(p)
            message += build_message_from_pick(pick) + "\n"
        except Exception:
            continue
    return send_telegram_message(message)

if __name__ == "__main__":
    ok = send_daily_picks()
    print("Sent:", ok)
