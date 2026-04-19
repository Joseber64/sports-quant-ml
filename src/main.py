import os
import requests
import pandas as pd
from datetime import datetime
import pathlib

# Crear carpeta data si no existe
pathlib.Path("data").mkdir(exist_ok=True)

def get_odds_data():
    api_key = os.getenv("ODDS_API_KEY")
    url = f"https://api.the-odds-api.com/v4/sports/upcoming/odds/?apiKey={api_key}&regions=us&markets=h2h"
    response = requests.get(url)
    try:
        data = response.json()
    except Exception:
        print("Error al decodificar Odds API:", response.text)
        return pd.DataFrame()
    if isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)

def save_csv(df, name):
    if df.empty:
        print(f"No se recibieron datos para {name}")
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/{name}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Guardado: {filename}")
    return filename

def main():
    print("Sports Quant ML conectado a APIs...")

    odds_df = get_odds_data()
    print("Odds API DataFrame:")
    print(odds_df.head())
    save_csv(odds_df, "odds")

if __name__ == "__main__":
    main()
