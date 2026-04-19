import os, requests, pandas as pd
from datetime import datetime
import pathlib

pathlib.Path("data").mkdir(exist_ok=True)

def fetch_odds():
    api_key = os.getenv("ODDS_API_KEY")
    sport = "soccer_epl"
    region = "uk"
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={api_key}&regions={region}&markets=h2h"

    response = requests.get(url)
    if response.status_code != 200:
        print("Error:", response.text)
        return None

    df = pd.json_normalize(response.json())
    filename = f"data/odds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"Datos guardados en {filename}")
    return filename

if __name__ == "__main__":
    fetch_odds()
