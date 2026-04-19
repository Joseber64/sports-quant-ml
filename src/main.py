import os
import requests
import pandas as pd

def get_odds_data():
    api_key = os.getenv("ODDS_API_KEY")
    url = f"https://api.the-odds-api.com/v4/sports/upcoming/odds/?apiKey={api_key}&regions=us&markets=h2h"
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data)

def get_all_sports_data():
    api_key = os.getenv("ALL_SPORTS_API_KEY")
    url = f"https://allsportsapi.com/api/football/?met=Fixtures&APIkey={api_key}"
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data.get("result", []))

def main():
    print("Sports Quant ML conectado a APIs...")

    # Obtener datos de Odds API
    odds_df = get_odds_data()
    print("Odds API DataFrame:")
    print(odds_df.head())

    # Obtener datos de All Sports API
    sports_df = get_all_sports_data()
    print("All Sports API DataFrame:")
    print(sports_df.head())

if __name__ == "__main__":
    main()
