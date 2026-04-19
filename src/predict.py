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
    home_team_name = str(row.get("HomeTeam", "")).lower()
    away_team_name = str(row.get("AwayTeam", "")).lower()

    if "bookmakers" in row and pd.notna(row["bookmakers"]):
        try:
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
                            elif name == home_team_name or home_team_name in name or name in home_team_name:
                                home = price
                            elif name == away_team_name or away_team_name in name or name in away_team_name:
                                away = price
                        
                        # Fallback extremo: si The Odds API mandó nombres muy distintos pero hay 3 cuotas, 
                        # sabemos que el formato estándar es (Local, Visita) excluyendo el "Draw".
                        if home is None and away is None and len(outcomes) >= 2:
                            draw_price = next((o["price"] for o in outcomes if str(o.get("name","")).lower() in ("draw","empate")), None)
                            remaining = [o["price"] for o in outcomes if str(o.get("name","")).lower() not in ("draw","empate")]
                            
                            if len(remaining) >= 2:
                                home, away = remaining[0], remaining[1]
                                draw = draw_price if draw_price else draw
                                
                        return home, draw, away
        except Exception as e:
            logging.error(f"Error al procesar bookmakers: {e}")

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
                "Prob_Local": round(prob_local, 3),
                "Prob_Visit": round(float(prob[0]), 3),
                "Poisson_2g_Local": round(poisson_example, 3),
                "Kelly_Fraction": round(kelly, 3)
            })
    else:
        print("No hay features disponibles o no hay modelo; intentando fallback con cuotas h2h...")
        for _, row in odds_df.iterrows():
            home = row.get("HomeTeam", "?")
            away = row.get("AwayTeam", "?")
            odd_home, odd_draw, odd_away = extract_h2h_odds(row)
            
            if odd_home is None and odd_away is None:
                continue
                
            p_home = implied_prob_from_odds(odd_home) if odd_home else None
            p_away = implied_prob_from_odds(odd_away) if odd_away else None
            
            if p_home and p_away:
                s = p_home + p_away + (1.0 / odd_draw if odd_draw else 0)
                if s > 0:
                    p_home /= s
                    p_away /= s
                    
            if p_home and p_away:
                pick = "Local gana" if p_home >= p_away else "Visita gana"
                prob_local = p_home
                kelly = kelly_fraction(prob_local, odds=odd_home if odd_home else 2.0)
                poisson_example = poisson_prob(lmbda=prob_local*2.0, k=2)
                results.append({
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "Prediction": pick,
                    "Prob_Local": round(prob_local, 3),
                    "Prob_Visit": round(p_away, 3),
                    "Poisson_2g_Local": round(poisson_example, 3),
                    "Kelly_Fraction": round(kelly, 3)
                })

    if not results:
        print("No se generaron predicciones (no había datos útiles).")
        return

    filename = f"data/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"Predicciones guardadas en {filename}")

if __name__ == "__main__":
    predict_new_matches()
