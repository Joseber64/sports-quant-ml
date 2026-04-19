import pandas as pd
import joblib
import glob
import pathlib
import math
from datetime import datetime

# Crear carpeta data si no existe
pathlib.Path("data").mkdir(exist_ok=True)

# --- Fórmulas profesionales ---
def poisson_prob(lmbda, k):
    """Probabilidad de marcar k goles dado lambda (media esperada)."""
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)

def kelly_fraction(prob, odds):
    """Fracción óptima de banca según Kelly Criterion."""
    b = odds - 1
    q = 1 - prob
    return (b * prob - q) / b if b > 0 else 0

# --- Cargar modelo entrenado ---
def load_model():
    return joblib.load("models/top5_leagues_model.pkl")

# --- Cargar último archivo de Odds API ---
def load_latest_odds():
    files = glob.glob("data/odds_*.csv")
    if not files:
        print("No se encontraron archivos de Odds API")
        return None
    latest = max(files)
    print(f"Usando archivo: {latest}")
    return pd.read_csv(latest)

# --- Predicciones ---
def predict_new_matches():
    model = load_model()
    odds_df = load_latest_odds()
    if odds_df is None:
        return

    # Features que entrenaste
    features = ["FTHG","FTAG","HS","AS","HST","AST","HC","AC","HF","AF","HY","AY","HR","AR"]
    available = [col for col in features if col in odds_df.columns]
    X = odds_df[available]
    teams = list(zip(odds_df.get("HomeTeam", ["?"]*len(odds_df)), odds_df.get("AwayTeam", ["?"]*len(odds_df))))

    preds = model.predict(X)
    probs = model.predict_proba(X)

    results = []
    for (home, away), p, prob in zip(teams, preds, probs):
        result = "Local gana" if p == 1 else "Visita gana"
        prob_local = prob[1]  # probabilidad de local
        prob_visit = prob[0]  # probabilidad de visita

        # Poisson: probabilidad de que el local meta 2 goles
        poisson_example = poisson_prob(lmbda=prob_local*2.0, k=2)

        # Kelly con cuota ejemplo 2.0
        kelly = kelly_fraction(prob_local, odds=2.0)

        print(f"{home} vs {away} → Predicción: {result}")
        print(f"Prob Local={prob_local:.2f}, Prob Visit={prob_visit:.2f}")
        print(f"Poisson(2 goles local)={poisson_example:.3f}, Kelly={kelly:.3f}\n")

        results.append({
            "HomeTeam": home,
            "AwayTeam": away,
            "Prediction": result,
            "Prob_Local": prob_local,
            "Prob_Visit": prob_visit,
            "Poisson_2g_Local": poisson_example,
            "Kelly_Fraction": kelly
        })

    # Guardar predicciones en CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/predictions_{timestamp}.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"Predicciones guardadas en {filename}")

if __name__ == "__main__":
    predict_new_matches()
