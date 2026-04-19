import os, requests, glob, pandas as pd

def send_telegram_message(message: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=data)
    print("Status:", response.status_code)
    print("Response:", response.text)

if __name__ == "__main__":
    files = glob.glob("data/predictions_*.csv")
    if not files:
        print("No hay predicciones para enviar")
    else:
        latest = max(files)
        df = pd.read_csv(latest)
        picks_text = "📊 Picks generados:\n"
        for _, row in df.iterrows():
            picks_text += f"{row['HomeTeam']} vs {row['AwayTeam']} → {row['Prediction']} (Prob Local={row['Prob_Local']:.2f}, Kelly={row['Kelly_Fraction']:.2f})\n"
        send_telegram_message(picks_text)
