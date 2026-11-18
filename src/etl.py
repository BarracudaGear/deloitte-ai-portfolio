"""ETL pipeline: clean Olist and PaySim CSVs and load into SQLite."""

import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("./data/deloitte_portfolio.db")
RAW_BASE = Path("./data/raw")

def clean_and_load(csv_path: Path, table_name: str, conn: sqlite3.Connection,
                   datetime_cols: list[str] | None = None,
                   dropna_subset: list[str] | None = None) -> int:
    """Load CSV, clean nulls and timestamps, write to SQLite, return row count."""
    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found, skipping {table_name}")
        return 0
    
    df = pd.read_csv(csv_path, low_memory=False)
    original_rows = len(df)
    
    # Drop rows that are entirely null or missing critical keys
    if dropna_subset:
        df = df.dropna(subset=dropna_subset)
    else:
        df = df.dropna(how="all")
    
    # Standardize timestamps
    if datetime_cols:
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Write to SQLite
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    final_rows = len(df)
    print(f"  {table_name}: {final_rows:,} rows (dropped {original_rows - final_rows:,})")
    return final_rows

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    
    print("Starting ETL into", DB_PATH)
    print("-" * 50)
    
    # Olist datasets
    olist_dir = RAW_BASE / "olist"
    olist_files = {
        "olist_orders_dataset.csv": {
            "table": "olist_orders",
            "datetime": ["order_purchase_timestamp", "order_approved_at",
                         "order_delivered_carrier_date", "order_delivered_customer_date",
                         "order_estimated_delivery_date"],
            "dropna": ["order_id"]
        },
        "olist_customers_dataset.csv": {
            "table": "olist_customers",
            "dropna": ["customer_id"]
        },
        "olist_order_items_dataset.csv": {
            "table": "olist_order_items",
            "datetime": ["shipping_limit_date"],
            "dropna": ["order_id"]
        },
        "olist_order_payments_dataset.csv": {
            "table": "olist_order_payments",
            "dropna": ["order_id"]
        },
        "olist_products_dataset.csv": {
            "table": "olist_products",
            "dropna": ["product_id"]
        },
        "olist_sellers_dataset.csv": {
            "table": "olist_sellers",
            "dropna": ["seller_id"]
        },
        "olist_order_reviews_dataset.csv": {
            "table": "olist_order_reviews",
            "datetime": ["review_creation_date", "review_answer_timestamp"],
            "dropna": ["review_id"]
        },
    }
    
    print("Loading Olist tables:")
    for filename, config in olist_files.items():
        clean_and_load(
            olist_dir / filename,
            config["table"],
            conn,
            datetime_cols=config.get("datetime"),
            dropna_subset=config.get("dropna")
        )
    
    # PaySim
    paysim_dir = RAW_BASE / "paysim"
    # Find the main CSV (usually one large file)
    paysim_csvs = list(paysim_dir.glob("*.csv"))
    if paysim_csvs:
        paysim_file = paysim_csvs[0]
        print(f"\nLoading PaySim from {paysim_file.name}:")
        df = pd.read_csv(paysim_file, low_memory=False)
        # Derive a simple timestamp from step (assuming 1 step = 1 hour from a base date)
        if "step" in df.columns:
            base = pd.Timestamp("2017-01-01")
            df["transaction_timestamp"] = base + pd.to_timedelta(df["step"], unit="h")
        df = df.dropna(subset=["amount"]) if "amount" in df.columns else df.dropna(how="all")
        df.to_sql("paysim_transactions", conn, if_exists="replace", index=False)
        print(f"  paysim_transactions: {len(df):,} rows")
    else:
        print("\nNo PaySim CSV found.")
    
    conn.close()
    print("-" * 50)
    print("ETL complete.")

if __name__ == "__main__":
    main()
