#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ---------------------------------------------------------
# Load the trained model + the exact feature list/order it expects
# ---------------------------------------------------------
model = joblib.load("random_forest_model.pkl")
model_features = joblib.load("model_features.pkl")

# Human-readable labels + help text for the form (keeps the HTML clean)
FEATURE_INFO = {
    "TERMINAL_ID_RISK_7DAY_WINDOW": {
        "label": "Terminal Risk (7-day)",
        "help": "Fraction of transactions at this terminal flagged as fraud in the last 7 days (0 to 1).",
        "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0
    },
    "TX_AMOUNT": {
        "label": "Transaction Amount ($)",
        "help": "Dollar amount of the current transaction.",
        "min": 0.0, "max": None, "step": 0.01, "default": 50.0
    },
    "TERMINAL_ID_RISK_30DAY_WINDOW": {
        "label": "Terminal Risk (30-day)",
        "help": "Fraction of transactions at this terminal flagged as fraud in the last 30 days (0 to 1).",
        "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0
    },
    "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW": {
        "label": "Customer Avg Amount (1-day)",
        "help": "This customer's average transaction amount over the last 1 day.",
        "min": 0.0, "max": None, "step": 0.01, "default": 50.0
    },
    "TERMINAL_ID_RISK_1DAY_WINDOW": {
        "label": "Terminal Risk (1-day)",
        "help": "Fraction of transactions at this terminal flagged as fraud in the last 1 day (0 to 1).",
        "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0
    },
    "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW": {
        "label": "Customer Avg Amount (7-day)",
        "help": "This customer's average transaction amount over the last 7 days.",
        "min": 0.0, "max": None, "step": 0.01, "default": 50.0
    },
}


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        features=model_features,
        feature_info=FEATURE_INFO,
        result=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    errors = []
    values = {}

    for feat in model_features:
        raw_value = request.form.get(feat, "")
        info = FEATURE_INFO[feat]
        try:
            val = float(raw_value)
        except ValueError:
            errors.append(f"{info['label']} must be a number.")
            continue

        if info["min"] is not None and val < info["min"]:
            errors.append(f"{info['label']} must be >= {info['min']}.")
        if info["max"] is not None and val > info["max"]:
            errors.append(f"{info['label']} must be <= {info['max']}.")

        values[feat] = val

    if errors:
        return render_template(
            "index.html",
            features=model_features,
            feature_info=FEATURE_INFO,
            result=None,
            errors=errors,
            submitted_values=request.form
        )

    # Build the feature vector in the EXACT order the model expects
    X = np.array([[values[feat] for feat in model_features]])

    proba = model.predict_proba(X)[0][1]  # probability of class 1 (fraud)
    percent = round(proba * 100, 2)

    if percent < 30:
        risk_level = "low"
    elif percent < 60:
        risk_level = "medium"
    else:
        risk_level = "high"

    result = {"percent": percent, "risk_level": risk_level}

    return render_template(
        "index.html",
        features=model_features,
        feature_info=FEATURE_INFO,
        result=result,
        submitted_values=request.form
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

