# src/predict.py
import pandas as pd
import joblib
import glob
import pathlib
import math
import logging
import ast
from datetime import datetime
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO)
pathlib.Path("data").mkdir(exist_ok=True)

def kelly_fraction(prob, odds):
    if odds is None or odds <= 1.0: return 0.0
    b = odds - 1
    q = 1 - prob
    return max(0.0, (b * prob - q) / b)

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
    return pd.read_csv(latest), latest

def implied_prob_from_odds(odds):
    return 1.0 / odds if odds and odds > 0 else None

def extract_match_odds(row):
    h_team = str(row.get("HomeTeam", row.get("home_team", ""))).lower()
    a_team = str(row.get("AwayTeam", row.get("away_team", ""))).lower()
    
    home, draw, away = None, None, None
    over_25, under_25 = None, None

    if "bookmakers" in row and pd.notna(row["bookmakers"]):
        try:
            bk = ast.literal_eval(str(row["bookmakers"]))
            if isinstance(bk, list) and len(bk) > 0:
                markets = bk[0].get("markets", [])
                for m in markets:
                    # Mercado 1X2
                    if m.get("key") in ("h2h", "h2h_lay"):
                        for o in m.get("outcomes", []):
                            name = str(o.get("name", "")).lower()
                            price = o.get("price")
                            if name in ("draw", "empate", "x"): draw = price
                            elif name == h_team or h_team in name: home = price
                            elif name == a_team or a_team in name: away = price
                    
                    # Mercado Totales (Goles)
                    elif m.get("key") == "totals":
                        for o in m.get("outcomes", []):
                            if o.get("point") == 2.5:
                                name = str(o.get("name", "")).lower()
                                if name == "over": over_25 = o.get("price")
                                elif name == "under": under_25 = o.get("price")
                                
        except Exception as e:
            logging.error(f"Error al procesar bookmakers: {e}")

    return home, draw, away, over_25, under_25

def predict_new_matches():
    model = None
    try:
        model = load_model()
    except Exception as e:
        print(f"Aviso: {e}. Se intentará fallback por cuotas.")

    odds_df, _ = load_latest_odds()
    if odds_df is None: return

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
            prob_main = float(prob[1]) if p == 1 else float(prob[0])
            results.append({
                "HomeTeam": home, "AwayTeam": away,
                "Prediction": "Local gana" if p == 1 else "Visita gana",
                "Prob_Main": round(prob_main, 3),
                "Kelly_Fraction": round(kelly_fraction(prob_main, 2.0), 3),
                "Over_2.5": None,
                "Under_2.5": None
            })
    else:
        print("Iniciando fallback por cuotas H2H y Totales...")
        for _, row in odds_df.iterrows():
            home = row.get("HomeTeam", row.get("home_team", "?"))
            away = row.get("AwayTeam", row.get("away_team", "?"))
            oh, od, oa, o25, u25 = extract_match_odds(row)
            
            if oh and od and oa:
                p_h = implied_prob_from_odds(oh)
                p_d = implied_prob_from_odds(od)
                p_a = implied_prob_from_odds(oa)
                
                # Normalización para eliminar el margen de la casa
                s = p_h + p_d + p_a
                p_h, p_d, p_a = p_h/s, p_d/s, p_a/s
                
                # Lógica para elegir Local, Empate o Visita
                if p_h >= p_a and p_h >= p_d:
                    pick, prob_main, odd_main = "Local gana", p_h, oh
                elif p_a >= p_h and p_a >= p_d:
                    pick, prob_main, odd_main = "Visita gana", p_a, oa
                else:
                    pick, prob_main, odd_main = "Empate", p_d, od

                results.append({
                    "HomeTeam": home, "AwayTeam": away,
                    "Prediction": pick,
                    "Prob_Main": round(prob_main, 3),
                    "Kelly_Fraction": round(kelly_fraction(prob_main, odd_main), 3),
                    "Over_2.5": o25,
                    "Under_2.5": u25
                })

    if results:
        filename = f"data/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(results).to_csv(filename, index=False)
        print(f"Archivo guardado: {filename}")

if __name__ == "__main__":
    predict_new_matches()
