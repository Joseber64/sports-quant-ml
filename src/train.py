import pandas as pd
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pathlib

pathlib.Path("models").mkdir(exist_ok=True)

def load_csvs():
    files = glob.glob("data/*.csv")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error leyendo {f}: {e}")
    return pd.concat(dfs, ignore_index=True)

def preprocess(df):
    df["result"] = df["FTR"].apply(lambda x: 1 if x == "H" else 0)
    features = ["FTHG","FTAG","HS","AS","HST","AST","HC","AC","HF","AF","HY","AY","HR","AR"]

    if "xG" in df.columns:
        features.append("xG")
    if "xGA" in df.columns:
        features.append("xGA")
    if "xG" in df.columns and "xGA" in df.columns:
        df["xG_diff"] = df["xG"] - df["xGA"]
        features.append("xG_diff")

    X = df[features]
    y = df["result"]
    return X, y

def train_model():
    df = load_csvs()
    print(f"Dataset combinado: {df.shape}")
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {acc:.2f}")

    joblib.dump(model, "models/top5_leagues_model.pkl")
    print("Modelo guardado en models/top5_leagues_model.pkl")

if __name__ == "__main__":
    train_model()
