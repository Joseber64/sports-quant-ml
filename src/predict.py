# src/predict.py
import pandas as pd
import joblib
import glob
import pathlib
import math
import json
import logging
import ast
from datetime import datetime
from sklearn.impute import SimpleImputer

# Configuración básica de logs
logging.basicConfig(level=logging.INFO)
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
    if odds is None or odds <= 0:
        return None
    return 1.0 / odds

def extract_h2h_odds(row):
    # Detecta nombres de columnas dinámicamente
    h_team = row.get("HomeTeam", row.get("home_team", ""))
    a_team = row.get("AwayTeam", row.get("away_team", ""))
    
    home_team_name = str(h_team).lower()
    away_team_name = str(a_team).lower()

    if "bookmakers" in row and pd.notna(row["bookmakers"]):
        try:
            # ast.literal_eval soluciona el error de comillas simples/dobles
            bk = ast.literal_eval(str(row["bookmakers"]))
            if isinstance(bk, list) and len(bk) > 0:
                markets = bk[0].get("markets", [])
                for m in markets:
                    if m.get("key") in ("h2h","h2h_lay"):
                        outcomes = m.get("outcomes", [])
                        home, draw, away = None, None, None
                        
                        for o in outcomes:
                            name = str(o.get("name", "")).lower()
                            price = o.get("price")
                            if name in ("draw", "empate", "x"):
                                draw = price
                            elif name == home_team_name or home_team_name in name:
                                home = price
                            elif name == away_team_name or away_team_name in name:
                                away = price
                        
                        return home, draw, away
        except Exception as e:
            logging.error(f"Error al procesar bookmakers: {e}")

    return None, None, None

def predict_new_matches():
    model = None
    try:
        model = load_model()
    except Exception as e:
        print(f"Aviso: {e}. Se intentará fallback por cuotas.")

    odds_df, odds_file = load_latest_odds()
    if odds_df is None: return

    # Intentar detectar features para el modelo
    features = ["FTHG","FTAG","HS","AS","HST","AST","HC","AC","HF","AF","HY","AY","HR","AR"]
    available = [c for c in features if c in odds_df.columns]
    
    results = []

    if model is not None and len(available) > 0:
        X = odds_df[available]
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=available)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        
        h_teams = odds_df.get("HomeTeam", odds_df.get("home_team", ["?"]*len(odds_df)))
        a_teams = odds_df.get("AwayTeam", odds_df.get("away_team", ["?"]*len(odds_df)))
        
        for (home, away), p, prob in zip(zip(h_teams, a_teams), preds, probs):
            prob_local = float(prob[1])
            results.append({
                "HomeTeam": home, "AwayTeam": away,
                "Prediction": "Local gana" if p == 1 else "Visita gana",
                "Prob_Local": round(prob_local, 3),
                "Kelly_Fraction": round(kelly_fraction(prob_local, 2.0), 3)
            })
    else:
        print("Iniciando fallback por cuotas H2H...")
        for _, row in odds_df.iterrows():
            home = row.get("HomeTeam", row.get("home_team", "?"))
            away = row.get("AwayTeam", row.get("away_team", "?"))
            oh, od, oa = extract_h2h_odds(row)
            
            if oh and oa:
                p_h = implied_prob_from_odds(oh)
                p_a = implied_prob_from_odds(oa)
                # Normalización básica
                s = p_h + p_a + (1.0/od if od else 0)
                p_h /= s
                results.append({
                    "HomeTeam": home, "AwayTeam": away,
                    "Prediction": "Local gana" if p_h >= (p_a/s) else "Visita gana",
                    "Prob_Local": round(p_h, 3),
                    "Kelly_Fraction": round(kelly_fraction(p_h, oh), 3)
                })

    if results:
        filename = f"data/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(results).to_csv(filename, index=False)
        print(f"Archivo guardado: {filename}")

if __name__ == "__main__":
    predict_new_matches()
