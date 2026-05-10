# 🏦 Bank Churn Predictor

Predict whether a bank customer will churn using XGBoost, with a Streamlit UI and a FastAPI REST endpoint.

**Model performance (test set, 10 000 rows):**

| Metric | Value |
|---|---|
| Accuracy | 81% |
| Recall — churn class | 74% |
| ROC-AUC | 0.86 |

Class imbalance (~80/20) is handled via `scale_pos_weight` in XGBoost.

---

## Project Structure

```
bank-churn-predictor/
├── data/
│   └── churn.csv                  # Raw dataset (10 000 rows)
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   └── 02_modeling.ipynb          # Model comparison & tuning
├── src/
│   ├── data_preprocessing.py      # Load, clean, encode, scale
│   ├── features.py                # Feature engineering
│   ├── train.py                   # XGBoost training + feature importance
│   └── evaluate.py                # Metrics + plots
├── models/
│   ├── churn_model.pkl            # Trained XGBoost model
│   └── scaler.pkl                 # Fitted StandardScaler
├── app/
│   ├── app.py                     # Streamlit UI (two-column layout)
│   └── api.py                     # FastAPI REST service
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── requirements.txt
├── pyproject.toml
├── main.py                        # Full training pipeline
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
# or, if using uv:
uv sync
```

---

## Usage

### 1. Train the model

```bash
python main.py
```

Runs the full pipeline:
- Loads and preprocesses `data/churn.csv`
- Engineers features (`balance_salary_ratio`, `age_group`, `zero_balance`)
- Trains XGBoost with class-imbalance handling (`scale_pos_weight`)
- Saves `models/churn_model.pkl` and `models/scaler.pkl`
- Saves evaluation plots + feature importance to `outputs/`

---

### 2. Launch the Streamlit app

```bash
streamlit run app/app.py
```

Two-column layout:
- Left: prediction result with colour-coded risk level + input summary
- Right: feature importance bar chart (always visible)

---

### 3. Start the FastAPI server

```bash
uvicorn app.api:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

#### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "credit_score": 620,
    "country": 1,
    "gender": 0,
    "age": 42,
    "tenure": 3,
    "balance": 95000,
    "products_number": 2,
    "credit_card": 1,
    "active_member": 0,
    "estimated_salary": 75000
  }'
```

#### Example response

```json
{
  "churn": 1,
  "churn_probability": 0.7312,
  "risk_level": "HIGH"
}
```

---

## Features

| Feature | Description |
|---|---|
| credit_score | Customer credit score |
| country | 0=France, 1=Germany, 2=Spain |
| gender | 0=Female, 1=Male |
| age | Customer age |
| tenure | Years with the bank |
| balance | Account balance |
| products_number | Number of bank products held |
| credit_card | Has credit card (0/1) |
| active_member | Is active member (0/1) |
| estimated_salary | Annual salary estimate |
| balance_salary_ratio | balance / salary *(engineered)* |
| age_group | Age bucket 0–3 *(engineered)* |
| zero_balance | Balance is zero flag *(engineered)* |

---

## Notebooks

| Notebook | Contents |
|---|---|
| `01_eda.ipynb` | Class distribution, correlations, distributions per feature |
| `02_modeling.ipynb` | Logistic Regression vs Random Forest vs XGBoost, hyperparameter tuning, model comparison table |
