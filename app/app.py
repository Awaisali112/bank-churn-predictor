"""Streamlit app for bank churn prediction."""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit as st

MODEL_PATH  = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"

FEATURE_NAMES = [
    "credit_score", "country", "gender", "age", "tenure",
    "balance", "products_number", "credit_card", "active_member",
    "estimated_salary", "balance_salary_ratio", "age_group", "zero_balance",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Fill in the customer details on the left and click **Predict** to see the churn risk.")

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("⚠️ Model artifacts not found. Run `python main.py` first.")
    st.stop()

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------------------------------------------------------------------
# Helper: feature importance chart (works for tree models + linear models)
# ---------------------------------------------------------------------------
def _render_importance(mdl, title: str):
    if hasattr(mdl, "feature_importances_"):
        importances = mdl.feature_importances_
        xlabel = "Importance (gain)"
    elif hasattr(mdl, "coef_"):
        importances = np.abs(mdl.coef_[0])
        xlabel = "|Coefficient|"
    else:
        st.info("Feature importance not available for this model type.")
        return

    indices      = np.argsort(importances)
    sorted_names = [FEATURE_NAMES[i] for i in indices]
    sorted_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(sorted_names, sorted_vals, color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sidebar — customer inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Customer Information")

credit_score     = st.sidebar.slider("Credit Score",        300,    850,    650)
age              = st.sidebar.slider("Age",                  18,    100,     35)
tenure           = st.sidebar.slider("Tenure (years)",        0,     10,      5)
balance          = st.sidebar.number_input("Balance ($)",    0.0, 300_000.0, 50_000.0, step=1000.0)
products_number  = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
credit_card      = st.sidebar.selectbox("Has Credit Card?",  ["Yes", "No"])
active_member    = st.sidebar.selectbox("Active Member?",    ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", 0.0, 250_000.0, 60_000.0, step=1000.0)
country          = st.sidebar.selectbox("Country",           ["France", "Germany", "Spain"])
gender           = st.sidebar.selectbox("Gender",            ["Female", "Male"])

# Encode
country_enc = {"France": 0, "Germany": 1, "Spain": 2}[country]
gender_enc  = 1 if gender == "Male" else 0
cc_enc      = 1 if credit_card == "Yes" else 0
am_enc      = 1 if active_member == "Yes" else 0

# Engineered features
balance_salary_ratio = balance / (estimated_salary + 1)
zero_balance = int(balance == 0)
if age <= 30:
    age_group = 0
elif age <= 45:
    age_group = 1
elif age <= 60:
    age_group = 2
else:
    age_group = 3

# ---------------------------------------------------------------------------
# Layout: two columns
# ---------------------------------------------------------------------------
col_result, col_importance = st.columns([1, 1])

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.sidebar.button("🔍 Predict Churn Risk", use_container_width=True):
    features = np.array([[
        credit_score, country_enc, gender_enc, age, tenure,
        balance, products_number, cc_enc, am_enc, estimated_salary,
        balance_salary_ratio, age_group, zero_balance,
    ]])
    features_scaled = scaler.transform(features)
    prediction  = model.predict(features_scaled)[0]
    probability = float(model.predict_proba(features_scaled)[0][1])

    with col_result:
        st.subheader("Prediction Result")

        # Colour-coded risk badge
        if probability >= 0.7:
            risk, colour = "HIGH", "🔴"
        elif probability >= 0.4:
            risk, colour = "MEDIUM", "🟡"
        else:
            risk, colour = "LOW", "🟢"

        st.metric("Churn Probability", f"{probability*100:.1f}%", delta=None)
        st.progress(probability)

        if prediction == 1:
            st.error(f"{colour} **{risk} CHURN RISK** — this customer is likely to leave.")
            st.markdown("**Recommendation:** Offer a retention incentive or loyalty programme.")
        else:
            st.success(f"{colour} **{risk} CHURN RISK** — this customer is likely to stay.")
            st.markdown("**Recommendation:** Continue standard service.")

        # Summary table
        st.divider()
        st.markdown("**Input summary**")
        summary = {
            "Credit Score": credit_score, "Age": age, "Tenure": tenure,
            "Balance ($)": f"{balance:,.0f}", "Products": products_number,
            "Credit Card": credit_card, "Active Member": active_member,
            "Salary ($)": f"{estimated_salary:,.0f}",
            "Country": country, "Gender": gender,
        }
        st.table(summary)

    with col_importance:
        st.subheader("Feature Importance")
        _render_importance(model, "What drives the prediction?")

else:
    with col_result:
        st.info("👈 Fill in the customer details and click **Predict Churn Risk**.")

    with col_importance:
        st.subheader("Feature Importance (global)")
        _render_importance(model, "Global Feature Importance")

# ---------------------------------------------------------------------------
# Model comparison (always shown at the bottom)
# ---------------------------------------------------------------------------
COMPARISON_IMG = "outputs/model_comparison.png"
if os.path.exists(COMPARISON_IMG):
    st.divider()
    st.subheader("📊 Model Selection — CV ROC-AUC Comparison")
    st.image(COMPARISON_IMG, use_container_width=True)
