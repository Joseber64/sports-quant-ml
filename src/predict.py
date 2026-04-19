import pandas as pd
import joblib
import glob
import pathlib
from datetime import datetime

# Crear carpeta data si no existe
pathlib.Path("data").mkdir(exist_ok=True)

def load_model():
    return joblib.load("models/top5_leagues_model.pkl")

def load_latest_odds():
    files = glob.glob("data/odds_*.csv")
    if not files:
        print("No se encontraron archivos de Odds API")
        return None
    latest = max(files)  # el más reciente
    print(f"Usando archivo: {latest}")
    return pd.read_csv(latest)

def preprocess_odds(df):
    # Aquí deberías mapear las columnas de Odds API a las mismas features que entrenaste
    # Ejemplo: si Odds API devuelve goles esperados, tiros, etc.
    # Ajusta según lo que realmente devuelva tu API
    features = ["FTHG","FTAG","HS","AS","HST","AST","HC","AC","HF","AF","HY","AY","HR","AR"]
    available = [col for col in features if col in df.columns]
    X = df[available]
    teams = list(zip(df.get("HomeTeam", ["?"]*len(df)), df.get("AwayTeam", ["?"]*len(df))))
    return X, teams

def predict_new_matches():
    model = load_model()
    odds_df = load_latest_odds()
    if odds_df is None:
        return
    X, teams = preprocess_odds(odds_df)
    preds = model.predict(X)

    results = []
    for (home, away), p in zip(teams, preds):
        result = "Local gana" if p == 1 else "Visita gana"
        print(f"{home} vs {away} → Predicción: {result}")
        results.append({"HomeTeam": home, "AwayTeam": away, "Prediction": result})

    # Guardar predicciones en CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/predictions_{timestamp}.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"Predicciones guardadas en {filename}")

if __name__ == "__main__":
    predict_new_matches()
