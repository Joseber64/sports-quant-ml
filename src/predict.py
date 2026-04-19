import pandas as pd
import joblib
import glob
import pathlib
import math
from datetime import datetime
from sklearn.impute import SimpleImputer

pathlib.Path("data").mkdir(exist_ok=True)

def poisson_prob(lmbda, k):
    return (math.exp(-lmbda) * (lmbda**k)) / math.factorial(k)

def kelly_fraction(prob, odds):
    b = odds - 1
    q = 1 - prob
    return (b * prob - q) / b if b > 0 else 0

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
        return None
    latest = max(files)
    print(f"Usando archivo: {latest}")
    return pd.read_csv(latest)

def predict_new_matches():
    model = load_model()
    odds_df = load_latest_odds()
    if odds_df is None:
        return

    features = ["FTHG","FTAG","HS","AS","HST","AST","HC","AC","HF","AF","HY","AY","HR","AR"]
    if "xG" in odds_df.columns:
        features.append("xG")
    if "xGA" in odds_df.columns:
        features.append("xGA")
    if "xG" in odds_df.columns and "xGA" in odds_df.columns:
        odds_df["xG_diff"] = odds_df["xG"] - odds_df["xGA"]
        features.append("xG_diff")

    available = [col for col in features if col in odds_df.columns]
    X = odds_df[available]

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=available)

    teams = list(zip(odds_df.get("HomeTeam", ["?"]*len(odds_df)), odds_df.get("AwayTeam", ["?"]*len(odds_df))))

    preds = model.predict(X)
    probs = model.predict_proba(X)

    results = []
    for (home, away), p, prob in zip(teams, preds, probs):
        result = "Local gana" if p == 1 else "Visita gana"
        prob_local = prob[1]
        prob_visit = prob[0]
        poisson_example = poisson_prob(lmbda=prob_local*2.0, k=2)
        kelly = kelly_fraction(prob_local, odds=2.0)

        results.append({
            "HomeTeam": home,
            "AwayTeam": away,
            "Prediction": result,
            "Prob_Local": prob_local,
            "Prob_Visit": prob_visit,
            "Poisson_2g_Local": poisson_example,
            "Kelly_Fraction": kelly
        })

    filename = f"data/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"Predicciones guardadas en {filename}")

if __name__ == "__main__":
    predict_new_matches()
