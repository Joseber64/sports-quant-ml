import os
import requests
import pandas as pd
from datetime import datetime
import pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

def train_model(df):
    # Simplificación: usamos las odds como features
    # Nota: la estructura real de la API es más compleja, aquí asumimos que hay campo 'bookmakers'
    if "bookmakers" not in df.columns:
        print("No hay datos suficientes para entrenar")
        return

    # Extraer odds de forma simplificada
    odds = []
    outcomes = []
    for row in df["bookmakers"]:
        if isinstance(row, list) and len(row) > 0:
            market = row[0].get("markets", [])
            if market:
                outcomes_list = market[0].get("outcomes", [])
                if len(outcomes_list) == 2:
                    odds.append([outcomes_list[0]["price"], outcomes_list[1]["price"]])
                    # Etiqueta ficticia: asumimos favorito como ganador (solo ejemplo)
                    outcomes.append(0 if outcomes_list[0]["price"] < outcomes_list[1]["price"] else 1)

    if not odds:
        print("No se pudieron extraer odds para entrenar")
        return

    X = pd.DataFrame(odds, columns=["team1_odds", "team2_odds"])
    y = pd.Series(outcomes)

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entrenar modelo
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluar
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {acc:.2f}")

def main():
    print("Sports Quant ML conectado a APIs...")

    # Odds API
    odds_df = get_odds_data()
    print("Odds API DataFrame:")
    print(odds_df.head())
    save_csv(odds_df, "odds")

    # Entrenar modelo con Odds API
    train_model(odds_df)

if __name__ == "__main__":
    main()
