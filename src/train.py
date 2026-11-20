import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.featurizer import calculate_all_rdkit_2d_descriptors
from src.splitter import scaffold_train_test_split


def main():
    data_path = os.path.join("data", "delaney.csv")
    model_dir = "models"
    model_path = os.path.join(model_dir, "rf_rdkit2d_model.joblib")

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    y = df["measured log(solubility:mol/L)"]

    print("Calculating full RDKit 2D descriptors...")
    X = calculate_all_rdkit_2d_descriptors(df["SMILES"])
    medians = X.median()
    X = X.fillna(medians)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_random = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_random.fit(X_tr, y_tr)
    y_pred_rand = rf_random.predict(X_te)
    print(f"[Random Split]   Test R²: {r2_score(y_te, y_pred_rand):.3f} | RMSE: {root_mean_squared_error(y_te, y_pred_rand):.3f}")

    X_tr_sc, X_te_sc, y_tr_sc, y_te_sc = scaffold_train_test_split(df, "SMILES", X, y, test_size=0.2)
    rf_scaff = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_scaff.fit(X_tr_sc, y_tr_sc)
    y_pred_scaff = rf_scaff.predict(X_te_sc)
    print(f"[Scaffold Split] Test R²: {r2_score(y_te_sc, y_pred_scaff):.3f} | RMSE: {root_mean_squared_error(y_te_sc, y_pred_scaff):.3f}")

    print("Training final model on full dataset...")
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X, y)

    os.makedirs(model_dir, exist_ok=True)
    artifact = {
        "model": final_model,
        "medians": medians,
        "feature_names": list(X.columns)
    }
    joblib.dump(artifact, model_path)
    print(f"Model saved successfully to {model_path}")


if __name__ == "__main__":
    main()
