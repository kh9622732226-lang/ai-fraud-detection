import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Payment Risk Intelligence", page_icon="🛡️", layout="wide")

MODEL_PATH = "models/fraud_model.joblib"
DATA_PATH = "data/demo_transactions.csv"

FEATURES = [
    "amount", "hour", "merchant_risk", "device_trust",
    "account_age_days", "transactions_24h", "failed_attempts_24h",
    "distance_from_usual_km", "is_new_device", "is_international",
    "payment_velocity"
]

def risk_factors(row):
    factors = []
    if row["amount"] >= 50000:
        factors.append(("High transaction amount", 24))
    if row["failed_attempts_24h"] >= 3:
        factors.append(("Multiple recent failed attempts", 20))
    if row["is_new_device"] == 1:
        factors.append(("New/untrusted device", 18))
    if row["is_international"] == 1:
        factors.append(("International transaction", 12))
    if row["distance_from_usual_km"] >= 300:
        factors.append(("Unusual transaction location", 15))
    if row["transactions_24h"] >= 15:
        factors.append(("Unusually high transaction frequency", 14))
    if row["merchant_risk"] >= 0.75:
        factors.append(("High-risk merchant profile", 16))
    if row["device_trust"] <= 0.30:
        factors.append(("Low device trust score", 14))
    return sorted(factors, key=lambda x: x[1], reverse=True)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"] if isinstance(bundle, dict) else bundle

st.title("🛡️ AI Payment Fraud Detection & Risk Intelligence")
st.caption("Machine-learning prototype for suspicious-transaction screening")

model = load_model()
if model is None:
    st.warning("Model not found. Run: `python src/generate_data.py` followed by `python src/train.py`.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔎 Single Transaction", "📊 Batch Analysis", "ℹ️ About"])

with tab1:
    st.subheader("Real-time transaction risk assessment")
    c1, c2, c3 = st.columns(3)
    with c1:
        amount = st.number_input("Amount (₹)", min_value=1.0, value=2500.0, step=500.0)
        hour = st.slider("Transaction hour", 0, 23, 14)
        merchant_risk = st.slider("Merchant risk", 0.0, 1.0, 0.25)
        device_trust = st.slider("Device trust", 0.0, 1.0, 0.85)
    with c2:
        account_age_days = st.number_input("Account age (days)", 1, 5000, 450)
        transactions_24h = st.number_input("Transactions in last 24h", 0, 100, 4)
        failed_attempts_24h = st.number_input("Failed attempts in last 24h", 0, 20, 0)
        distance = st.number_input("Distance from usual location (km)", 0.0, 5000.0, 5.0)
    with c3:
        is_new_device = st.checkbox("New device")
        is_international = st.checkbox("International transaction")
        payment_velocity = st.number_input("Payment velocity (transactions/hour)", 0.0, 50.0, 1.0)

    row = pd.DataFrame([{
        "amount": amount, "hour": hour, "merchant_risk": merchant_risk,
        "device_trust": device_trust, "account_age_days": account_age_days,
        "transactions_24h": transactions_24h, "failed_attempts_24h": failed_attempts_24h,
        "distance_from_usual_km": distance, "is_new_device": int(is_new_device),
        "is_international": int(is_international), "payment_velocity": payment_velocity
    }])

    if st.button("Assess Risk", type="primary", use_container_width=True):
        probability = float(model.predict_proba(row[FEATURES])[0, 1])
        score = round(probability * 100, 1)

        if score >= 75:
            label, action = "HIGH RISK", "Review / step-up verification"
        elif score >= 40:
            label, action = "MEDIUM RISK", "Additional verification recommended"
        else:
            label, action = "LOW RISK", "Allow, subject to normal controls"

        a, b, c = st.columns(3)
        a.metric("Fraud probability", f"{score}%")
        b.metric("Risk level", label)
        c.metric("Suggested action", action)

        st.progress(min(score / 100, 1.0))
        st.subheader("Why was this transaction flagged?")
        factors = risk_factors(row.iloc[0])
        if factors:
            for name, weight in factors[:5]:
                st.write(f"• **{name}** — risk contribution: {weight}")
        else:
            st.success("No major rule-based warning factors detected.")

with tab2:
    st.subheader("Batch transaction screening")
    uploaded = st.file_uploader("Upload a CSV with the required transaction columns", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        missing = [x for x in FEATURES if x not in df.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            probs = model.predict_proba(df[FEATURES])[:, 1]
            result = df.copy()
            result["fraud_probability"] = np.round(probs, 4)
            result["risk_score"] = np.round(probs * 100, 1)
            result["risk_level"] = pd.cut(
                result["risk_score"], [-1, 39.99, 74.99, 100],
                labels=["LOW", "MEDIUM", "HIGH"]
            )
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Download scored CSV",
                result.to_csv(index=False).encode(),
                "scored_transactions.csv",
                "text/csv"
            )
            st.bar_chart(result["risk_level"].value_counts())

    elif os.path.exists(DATA_PATH):
        st.info("Tip: generate demo data and upload `data/demo_transactions.csv` to test batch scoring.")

with tab3:
    st.markdown("""
### What this project demonstrates
- Supervised machine learning for fraud-risk classification
- Imbalanced-data handling through cost-sensitive training
- Feature engineering from transaction behaviour
- Explainable risk indicators
- Interactive model serving
- Batch inference and downloadable results

### Safety & scope
The project uses synthetic data. It is a demonstration, not a production payment-security system.
""")
