import pandas as pd
import joblib
import glob

def load_model():
    return joblib.load("models/top5_leagues_model.pkl")

def load_latest_odds():
    # Busca el último archivo odds generado por main.py
    files = glob.glob("data/odds_*.csv")
    if not files:
        print("No se encontraron archivos de Odds API")
        return None
    latest = max(files)  # el más reciente
    print(f"Usando archivo: {latest}")
    return pd.read_csv(latest)

def preprocess_odds(df):
    # Extraer features similares a los históricos
    # Aquí asumimos que Odds API devuelve cuotas y mercados
    # Simplificación: usamos precios como proxies de fuerza
    odds_features = []
    teams = []
    for i, row in df.iterrows():
        if "bookmakers" in df.columns:
            # Si la API devuelve estructura compleja, habría que normalizar
            continue
        # Ejemplo: si ya tienes columnas de goles/tiros en odds_df
        if {"FTHG","FTAG","HS","AS","HST","AST"}.issubset(df.columns):
            odds_features.append([row["FTHG"], row["FTAG"], row["HS"], row["AS"], row["HST"], row["AST"]])
            teams.append((row["HomeTeam"], row["AwayTeam"]))
    return pd.DataFrame(odds_features, columns=["FTHG","FTAG","HS","AS","HST","AST"]), teams

def predict_new_matches():
    model = load_model()
    odds_df = load_latest_odds()
    if odds_df is None:
        return
    X, teams = preprocess_odds(odds_df)
    preds = model.predict(X)
    for (home, away), p in zip(teams, preds):
        result = "Local gana" if p == 1 else "Visita gana"
        print(f"{home} vs {away} → Predicción: {result}")

if __name__ == "__main__":
    predict_new_matches()
