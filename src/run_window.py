# src/run_window.py
"""
Script entrypoint used by CI. Detecta la ventana actual (según hora UTC)
y ejecuta predict + send_telegram para ventanas 08:00, 12:00, 16:00 CST (14:00,18:00,22:00 UTC).
A las 05:00 UTC (23:00 CST) ejecuta cierre de jornada.
"""
import sys
from datetime import datetime, timezone
from src.predict import predict_and_save
from src.send_telegram import send_daily_picks
from src.close_day import close_day_process

def main():
    now = datetime.now(timezone.utc)
    hour = now.hour
    # Ventanas en UTC: 14, 18, 22 -> correspond to 08,12,16 CST
    if hour == 14 or hour == 18 or hour == 22:
        window_label = {14: "08", 18: "12", 22: "16"}[hour]
        print(f"Running prediction window {window_label} (UTC hour {hour})")
        picks = predict_and_save(window_label=window_label)
        if picks:
            print(f"Generated {len(picks)} picks, sending to Telegram...")
            send_daily_picks()
        else:
            print("No new picks generated.")
    # Cierre de jornada a las 05:00 UTC (23:00 CST)
    elif hour == 5:
        print("Running end-of-day close process (05:00 UTC / 23:00 CST)")
        close_day_process()
    else:
        # If triggered manually, run predict+send
        print("Manual or unscheduled run: running predict and send")
        picks = predict_and_save(window_label="manual")
        if picks:
            send_daily_picks()
        else:
            print("No picks generated on manual run.")

if __name__ == "__main__":
    main()
