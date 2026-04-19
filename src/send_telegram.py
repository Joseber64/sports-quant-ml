# src/send_telegram.py
import os
import requests
import glob
import pandas as pd
from datetime import datetime

def send_telegram_message(message: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID en el entorno.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    response = requests.post(url, data=data, timeout=15)
    print("Status:", response.status_code)
    print("Response:", response.text)

if __name__ == "__main__":
    files = glob.glob("data/predictions_*.csv")
    if not files:
        print("No hay predicciones para enviar")
    else:
        latest = max(files)
        df = pd.read_csv(latest)
        if df.empty:
            print("El archivo de predicciones está vacío:", latest)
        else:
            # Construir mensaje con formato limpio (Markdown)
            picks_text = f"*📊 Picks generados — {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
            for _, row in df.iterrows():
                home = row.get("HomeTeam", "?")
                away = row.get("AwayTeam", "?")
                pred = row.get("Prediction", "?")
                prob_local = row.get("Prob_Local", None)
                kelly = row.get("Kelly_Fraction", None)
                picks_text += f"*{home}* vs *{away}* — {pred}\n"
                if prob_local is not None:
                    picks_text += f"Prob Local: `{prob_local:.2f}`"
                if kelly is not None:
                    picks_text += f"  •  Kelly: `{kelly:.2f}`"
                picks_text += "\n\n"
            send_telegram_message(picks_text)
