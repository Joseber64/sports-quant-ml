# src/predict.py
import pandas as pd
import joblib
import glob
import pathlib
import math
from datetime import datetime
from sklearn.impute import SimpleImputer
import json

pathlib.Path("data").mkdir(exist_ok=True)

def poisson_prob(lmbda, k):
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)

def kelly_fraction(prob, odds):
    b = odds - 1
    q = 1 - prob
    return max(0.0, (b * prob - q) / b) if b > 0 else 0.0

def load_model():
    files = glob.glob("models/top5_leagues_model_*.pkl")
    if not files:
        raise FileNotFoundError("No se encontró ningún modelo entrenado")
    latest = max(files)
    print(f"Usando modelo: {latest}")
    return joblib.load(latest)

def load_latest_odds():
    files = glob.glob("data/odds_*.csv")
    if not files:
        print("No se encontraron archivos de Odds API")
        return None, None
    latest = max(files)
    print(f"Usando archivo: {latest}")
    df = pd.read_csv(latest)
    return df, latest

def implied_prob_from_odds(odds):
    # odds en formato decimal
    if odds is None or odds <= 0:
        return None
    return 1.0 / odds

def extract_h2h_odds(row):
    # Intenta extraer cuotas h2h desde columnas comunes (ajusta según tu CSV)
    # Si tu CSV tiene una columna 'bookmakers' con JSON, intenta parsearla.
    # Devuelve (odd_home, odd_draw, odd_away) o (None,None,None)
    # Ejemplo: si hay columna 'bookmakers' con JSON:
    if "bookmakers" in row and pd.notna(row["bookmakers"]):
        try:
            bk = json.loads(row["bookmakers"])
            if isinstance(bk, list) and len(bk) > 0:
                markets = bk[0].get("markets", [])
                for m in markets:
                    if m.get("key") in ("h2h","h2h_lay"):
                        outcomes = m.get("outcomes", [])
                        # outcomes: [{'name':'Home','price':1.9}, ...]
                        home = next((o["price"] for o in outcomes if o["name"].lower() in ("home","local")), None)
                        away = next((o["price"] for o in outcomes if o["name"].lower() in ("away","visitante","away")), None)
                        draw = next((o["price"] for o in outcomes if o["name"].lower() in ("draw","empate")), None)
                        return home, draw, away
        try:
    do_something()
except ValueError:
    handle_value_error()
except Exception as e:
    log_error(e)
    # Si hay columnas directas
    for colset in [("home_odds","draw_odds","away_odds"), ("h2h_home","h2h_draw","h2h_away")]:
        if all(c in row.index for c in colset):
            return row[colset[0]], row[colset[1]], row[colset[2]]
    return None, None, None

def predict_new_matches():
    model = None
    try:
        model = load_model()
    except FileNotFoundError:
        print("No hay modelo entrenado, se usará fallback por cuotas si es posible.")

    odds_df, odds_file = load_latest_odds()
    if odds_df is None:
        return

    # Features esperadas por el modelo
    features = ["FTHG","FTAG","HS","AS","HST","AST","HC","AC","HF","AF","HY","AY","HR","AR"]
    if "xG" in odds_df.columns:
        features.append("xG")
    if "xGA" in odds_df.columns:
        features.append("xGA")
    if "xG" in odds_df.columns and "xGA" in odds_df.columns:
        odds_df["xG_diff"] = odds_df["xG"] - odds_df["xGA"]
        features.append("xG_diff")

    available = [c for c in features if c in odds_df.columns]
    print(f"Features shape: {odds_df.shape}, Available features: {len(available)}")

    results = []

    if model is not None and len(available) > 0:
        X = odds_df[available]
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=available)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        teams = list(zip(odds_df.get("HomeTeam", ["?"]*len(odds_df)), odds_df.get("AwayTeam", ["?"]*len(odds_df))))
        for (home, away), p, prob in zip(teams, preds, probs):
            prob_local = float(prob[1])
            poisson_example = poisson_prob(lmbda=prob_local*2.0, k=2)
            kelly = kelly_fraction(prob_local, odds=2.0)
            results.append({
                "HomeTeam": home,
                "AwayTeam": away,
                "Prediction": "Local gana" if p == 1 else "Visita gana",
                "Prob_Local": prob_local,
                "Prob_Visit": float(prob[0]),
                "Poisson_2g_Local": poisson_example,
                "Kelly_Fraction": kelly
            })
    else:
        # Fallback: generar picks desde cuotas h2h si existen
        print("No hay features disponibles o no hay modelo; intentando fallback con cuotas h2h...")
        for _, row in odds_df.iterrows():
            home = row.get("HomeTeam", "?")
            away = row.get("AwayTeam", "?")
            odd_home, odd_draw, odd_away = extract_h2h_odds(row)
            if odd_home is None and odd_away is None:
                continue
            p_home = implied_prob_from_odds(odd_home) if odd_home else None
            p_away = implied_prob_from_odds(odd_away) if odd_away else None
            # Normalizar si ambos existen
            if p_home and p_away:
                s = p_home + p_away + (1.0 / odd_draw if odd_draw else 0)
                if s > 0:
                    p_home /= s
                    p_away /= s
            # Elegir pick
            if p_home and p_away:
                pick = "Local gana" if p_home >= p_away else "Visita gana"
                prob_local = p_home
                kelly = kelly_fraction(prob_local, odds=odd_home if odd_home else 2.0)
                poisson_example = poisson_prob(lmbda=prob_local*2.0, k=2)
                results.append({
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "Prediction": pick,
                    "Prob_Local": prob_local,
                    "Prob_Visit": p_away,
                    "Poisson_2g_Local": poisson_example,
                    "Kelly_Fraction": kelly
                })

    if not results:
        print("No se generaron predicciones (no había datos útiles).")
        return

    filename = f"data/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"Predicciones guardadas en {filename}")

if __name__ == "__main__":
    predict_new_matches()
