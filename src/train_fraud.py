"""Train an IsolationForest anomaly detector on a 200k sample of PaySim transactions."""

import sqlite3
from pathlib import Path
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

DB_PATH = Path("./data/deloitte_portfolio.db")
MODEL_DIR = Path("./models")
MODEL_PATH = MODEL_DIR / "fraud_iforest.joblib"
SAMPLE_SIZE = 200_000

def load_sample() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    # Stratified-ish sample: take all fraud + random non-fraud to reach ~200k
    fraud_df = pd.read_sql(
        "SELECT * FROM paysim_transactions WHERE isFraud = 1",
        conn,
    )
    non_fraud_limit = max(0, SAMPLE_SIZE - len(fraud_df))
    non_fraud_df = pd.read_sql(
        f"SELECT * FROM paysim_transactions WHERE isFraud = 0 ORDER BY RANDOM() LIMIT {non_fraud_limit}",
        conn,
    )
    conn.close()
    df = pd.concat([fraud_df, non_fraud_df], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["delta_dest"] = df["oldbalanceDest"] - df["newbalanceDest"]
    df["amount_to_orig"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amount_to_dest"] = df["amount"] / (df["oldbalanceDest"] + 1)
    return df

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("fraud_anomaly")

    print("Loading PaySim sample...")
    df = load_sample()
    print(f"Sample size: {len(df):,} (fraud rate: {df['isFraud'].mean():.4%})")

    df = engineer_features(df)
    feature_cols = [
        "amount", "delta_orig", "delta_dest",
        "amount_to_orig", "amount_to_dest",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
    ]
    # Keep only numeric features that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df["isFraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="iforest_paysim"):
        model = IsolationForest(
            n_estimators=100,
            contamination=0.0015,  # approximate fraud rate
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train)

        # IsolationForest: -1 = anomaly, 1 = normal → map to is_anomaly
        raw_preds = model.predict(X_test)
        is_anomaly = (raw_preds == -1).astype(int)

        precision = precision_score(y_test, is_anomaly, zero_division=0)
        recall = recall_score(y_test, is_anomaly, zero_division=0)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("contamination", 0.0015)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        # mlflow.sklearn.log_model(model, "model")

        print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")

        artifact = {"model": model, "feature_cols": feature_cols}
        joblib.dump(artifact, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
