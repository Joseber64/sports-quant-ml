import pandas as pd
import glob
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Crear carpeta models si no existe
pathlib.Path("models").mkdir(exist_ok=True)

def load_data():
    # Cargar todos los CSV de la carpeta data
    files = glob.glob("data/*.csv")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df

def preprocess(df):
    # Quitamos empates y creamos columna objetivo
    df = df[df["FTR"] != "D"]
    df["result"] = df["FTR"].apply(lambda x: 1 if x == "H" else 0)

    # Features simples: goles y tiros
    X = df[["FTHG", "FTAG", "HS", "AS", "HST", "AST"]]
    y = df["result"]
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {acc:.2f}")

    # Guardar modelo
    joblib.dump(model, "models/top5_leagues_model.pkl")
    print("Modelo guardado en models/top5_leagues_model.pkl")

def main():
    df = load_data()
    print("Dataset combinado:", df.shape)
    X, y = preprocess(df)
    train_model(X, y)

if __name__ == "__main__":
    main()
