# AI-Powered Payment Fraud Detection & Risk Intelligence

An advanced, competition-ready prototype for detecting suspicious digital-payment transactions using machine learning.

## Highlights
- Fraud probability + human-readable risk score (0–100)
- Explainable risk factors for every transaction
- Batch CSV scoring
- Interactive Streamlit dashboard
- Synthetic-data generator for safe demo/testing
- Time-aware feature engineering
- Cost-sensitive model training for highly imbalanced fraud data
- Precision, recall, F1, PR-AUC and confusion-matrix evaluation
- No real customer/payment data is included

## Architecture

Transaction data → validation → feature engineering → ML risk model → probability calibration → risk score → explanations → dashboard

## Quick start

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
streamlit run app.py
```

The app will create/use `models/fraud_model.joblib`.

## Demo features

The dashboard supports:
1. Single-transaction risk assessment
2. Batch CSV scoring
3. Risk distribution and fraud-rate analytics
4. Top risk factors

## CSV columns

`amount, hour, merchant_risk, device_trust, account_age_days, transactions_24h, failed_attempts_24h, distance_from_usual_km, is_new_device, is_international, payment_velocity`

## Important

This is a prototype for an AI-builder/hackathon submission. It does **not** connect to Razorpay production systems and should not be used for real payment decisions without proper validation, security review, monitoring, privacy controls and compliance.
