import pandas as pd
import joblib

def load_model():
    return joblib.load("models/top5_leagues_model.pkl")

def predict_new_data(new_csv):
    df = pd.read_csv(new_csv)
    if "FTR" in df.columns:
        df = df[df["FTR"] != "D"]
    X = df[["FTHG", "FTAG", "HS", "AS", "HST", "AST"]]
    model = load_model()
    predictions = model.predict(X)
    df["prediction"] = predictions
    print(df[["HomeTeam", "AwayTeam", "prediction"]].head())
    return df

if __name__ == "__main__":
    predict_new_data("data/odds_latest.csv")  # ejemplo con el archivo generado por main.py
