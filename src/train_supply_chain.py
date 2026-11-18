"""Train an XGBoost demand forecaster on weekly Olist product-category demand."""

import sqlite3
from pathlib import Path
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

DB_PATH = Path("./data/deloitte_portfolio.db")
MODEL_DIR = Path("./models")
MODEL_PATH = MODEL_DIR / "supply_chain_xgb.joblib"

def load_and_aggregate() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        DATE(o.order_purchase_timestamp, 'start of day', '-' || strftime('%w', o.order_purchase_timestamp) || ' days') AS week_start,
        COALESCE(p.product_category_name, 'unknown') AS product_category,
        COUNT(DISTINCT o.order_id) AS order_count,
        SUM(oi.price) AS total_revenue,
        COUNT(*) AS item_count
    FROM olist_orders o
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    LEFT JOIN olist_products p ON oi.product_id = p.product_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
    GROUP BY week_start, product_category
    ORDER BY week_start, product_category
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_category", "week_start"]).reset_index(drop=True)
    for lag in [1, 2, 4]:
        df[f"lag_{lag}_orders"] = df.groupby("product_category")["order_count"].shift(lag)
    df["rolling_4_mean"] = (
        df.groupby("product_category")["order_count"]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    )
    df["rolling_4_std"] = (
        df.groupby("product_category")["order_count"]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).std())
    )
    df = df.dropna(subset=["lag_1_orders", "lag_2_orders", "lag_4_orders"]).reset_index(drop=True)
    return df

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("supply_chain_demand")

    print("Loading and aggregating Olist data...")
    raw = load_and_aggregate()
    print(f"Aggregated rows: {len(raw):,}")

    df = engineer_features(raw)
    print(f"Rows after feature engineering: {len(df):,}")

    feature_cols = ["lag_1_orders", "lag_2_orders", "lag_4_orders", "rolling_4_mean", "rolling_4_std"]
    X = df[feature_cols]
    y = df["order_count"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="xgb_weekly_demand"):
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        # mlflow.sklearn.log_model(model, "model")

        print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f}")

        # Persist model + feature list
        artifact = {"model": model, "feature_cols": feature_cols}
        joblib.dump(artifact, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
