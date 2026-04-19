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
            picks_text = f"*📊 Picks Actualizados — {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
            for _, row in df.iterrows():
                home = row.get("HomeTeam", "?")
                away = row.get("AwayTeam", "?")
                pred = row.get("Prediction", "?")
                prob_main = row.get("Prob_Main", None)
                kelly = row.get("Kelly_Fraction", None)
                over_25 = row.get("Over_2.5", None)
                under_25 = row.get("Under_2.5", None)
                
                picks_text += f"⚽ *{home}* vs *{away}*\n"
                picks_text += f"🎯 Pick: *{pred}*\n"
                
                if pd.notna(prob_main):
                    picks_text += f"Probabilidad: `{prob_main:.2f}`"
                if pd.notna(kelly) and kelly > 0:
                    picks_text += f" | Kelly: `{kelly:.2f}`"
                picks_text += "\n"
                
                if pd.notna(over_25) and pd.notna(under_25):
                    picks_text += f"🥅 Goles 2.5 -> Más: `{over_25}` | Menos: `{under_25}`\n"
                    
                picks_text += "\n"
                
            send_telegram_message(picks_text)
