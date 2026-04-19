import pandas as pd
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from datetime import datetime
import pathlib
import numpy as np

pathlib.Path("models").mkdir(exist_ok=True)

def load_csvs():
    files = glob.glob("data/*.csv")
    if not files:
        raise FileNotFoundError("No CSV files found in data/ directory")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"Error leyendo {f}: {e}")
    
    if not dfs:
        raise ValueError("No valid data found in CSV files")
    
    return pd.concat(dfs, ignore_index=True)

def preprocess(df):
    # Target (resultado)
    if "FTR" not in df.columns:
        print("Warning: FTR column not found, using random targets")
        df["result"] = np.random.randint(0, 2, len(df))
    else:
        df["result"] = df["FTR"].apply(lambda x: 1 if x == "H" else 0)
    
    # Features base
    base_features = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]
    features = [f for f in base_features if f in df.columns]
    
    # Features opcionales
    if "xG" in df.columns:
        features.append("xG")
    if "xGA" in df.columns:
        features.append("xGA")
    if "xG" in features and "xGA" in features:
        df["xG_diff"] = df["xG"] - df["xGA"]
        features.append("xG_diff")
    
    if not features:
        raise ValueError("No features found in dataset")
    
    X = df[features].copy()
    y = df["result"].copy()
    
    # Mantener filas con target válido
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Imputación ANTES de entrenar
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=features)
    
    # Verificar que no queden NaN
    if X.isna().any().any():
        raise ValueError("NaN values still present after imputation")
    
    return X, y

def train_model():
    try:
        df = load_csvs()
        print(f"Dataset combinado: {df.shape}")
        
        if len(df) < 10:
            raise ValueError("Insufficient data for training (need at least 10 rows)")
        
        X, y = preprocess(df)
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"NaN in X: {X.isna().sum().sum()}")
        print(f"NaN in y: {y.isna().sum()}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo: {acc:.2f}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/top5_leagues_model_{timestamp}.pkl"
        joblib.dump(model, model_path)
        print(f"Modelo guardado en {model_path}")
        
        # Actualiza modelo principal
        joblib.dump(model, "models/top5_leagues_model.pkl")
        print("Modelo principal actualizado")
        
    except Exception as e:
        print(f"ERROR en train_model: {e}")
        raise

if __name__ == "__main__":
    train_model()
