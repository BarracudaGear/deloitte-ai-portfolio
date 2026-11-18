# Deloitte AI Portfolio – 5-Project Tracker

**Owner:** NS DataLab / BarracudaGear  
**Purpose:** Structured tracking document for a 5-project AI portfolio demonstrating end-to-end capabilities in e-commerce analytics and financial fraud detection, targeted for Deloitte AI roles.

---

## Directory Structure

```
deloitte-ai-portfolio/
├── data/
│   ├── raw/
│   │   ├── olist/          # Brazilian E-Commerce CSVs
│   │   └── paysim/         # PaySim fraud transaction CSVs
│   └── deloitte_portfolio.db
├── scripts/
│   └── download_data.py
├── src/
│   └── etl.py
├── notebooks/            # Exploratory analysis & modeling
├── models/               # Trained model artifacts
├── dashboards/           # Streamlit / FastAPI apps
├── tests/
├── JobHunt.md
├── requirements.txt
└── .gitignore
```

---

## SQLite Schemas

Database: `./data/deloitte_portfolio.db`

### Olist Tables
- **olist_orders**: order_id (PK), customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date
- **olist_customers**: customer_id (PK), customer_unique_id, customer_zip_code_prefix, customer_city, customer_state
- **olist_order_items**: order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value
- **olist_order_payments**: order_id, payment_sequential, payment_type, payment_installments, payment_value
- **olist_products**: product_id (PK), product_category_name, product_name_length, product_description_length, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm
- **olist_sellers**: seller_id (PK), seller_zip_code_prefix, seller_city, seller_state
- **olist_order_reviews**: review_id (PK), order_id, review_score, review_comment_title, review_comment_message, review_creation_date, review_answer_timestamp

### PaySim Tables
- **paysim_transactions**: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud, transaction_timestamp (derived)

All timestamp columns are standardized to ISO datetime format. Nulls cleaned on critical keys.

---

## Dataset Sources

| Dataset | Kaggle Handle | Description | License / Notes |
|---------|---------------|-------------|-----------------|
| Brazilian E-Commerce | `olistbr/brazilian-ecommerce` | Real commercial data from Brazilian e-commerce (orders, customers, products, payments, reviews, sellers, geolocation). ~100k orders. | Public, for research and portfolio use. |
| PaySim Synthetic Financial Dataset | `ealaxi/paysim1` | Synthetic mobile money transaction data generated from real financial logs. Used for fraud detection research. Contains isFraud labels. | Public synthetic dataset. |

**Download:** Run `python scripts/download_data.py` (requires Kaggle credentials configured for kagglehub).

---

## 5 Portfolio Projects

1. **Customer Lifetime Value (CLV) Prediction** – Olist  
   Predict future customer value using RFM features + gradient boosting (XGBoost).

2. **Product Recommendation Engine** – Olist  
   Collaborative filtering / content-based recommendations for cross-sell.

3. **Delivery Performance Forecasting** – Olist  
   Predict on-time delivery probability and identify delay drivers.

4. **Mobile Money Fraud Detection** – PaySim  
   Binary classification of fraudulent transactions with feature engineering on balances and transaction types. Focus on precision/recall for rare events.

5. **Integrated Risk & Revenue Analytics Dashboard** – Combined  
   Streamlit + FastAPI application surfacing key metrics, model scores, and interactive filters across both domains.

---

## Daily Changelog

### 2026-11-18
- Initial repository setup on `main`.
- Added `.gitignore`, `requirements.txt`.
- Created data download script (`scripts/download_data.py`) for Olist and PaySim.
- Created ETL script (`src/etl.py`) to clean and load CSVs into SQLite.
- Established tracking document structure for the 5-project portfolio.

---

*Last updated: 2026-11-18*
