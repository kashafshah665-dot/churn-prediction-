from flask import Flask, request, jsonify, render_template
import pandas as pd
from model_loader import load_model

app = Flask(__name__)

# =========================
# CONFIGURABLE THRESHOLDS
# =========================
LOW_RISK_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.48

# =========================
# LOAD MODEL
# =========================
model, scaler = load_model()

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("gym_membership_renewal.csv")

# =========================
# PREPROCESS ENTIRE DATASET
# =========================

# Encode gender properly
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

feature_columns = [
    "age",
    "gender",
    "subscription_length",
    "membership_type",
    "num_logins",
    "num_complaints",
    "num_classes_attended",
    "avg_session_time"
]

X = df[feature_columns]

# Scale
X_scaled = scaler.transform(X)

# Batch prediction (FAST)
df["predicted_churn"] = model.predict(X_scaled)
df["predicted_probability"] = model.predict_proba(X_scaled)[:, 1]

# =========================
# RISK LEVEL LOGIC
# =========================
def assign_risk(prob):
    if prob < LOW_RISK_THRESHOLD:
        return "Low"
    elif prob < HIGH_RISK_THRESHOLD:
        return "Medium"
    else:
        return "High"

df["risk_level"] = df["predicted_probability"].apply(assign_risk)


# =========================
# SUGGESTION LOGIC
# =========================
def generate_suggestion(row):
    if row["num_complaints"] >= 4:
        return "Customer Support Follow-up Recommended"
    elif row["num_logins"] <= 3:
        return "Member Re-engagement Recommended"
    elif row["subscription_length"] <= 3:
        return "Long-term Plan Incentive Recommended"
    else:
        return "Retention Offer Recommended"

df["suggested_action"] = df.apply(generate_suggestion, axis=1)

# =========================
# MODEL EVALUATION (TEST SPLIT)
# =========================

from sklearn.model_selection import train_test_split

y = df["renewed_membership"]
X = df[feature_columns]

# Scale again (same scaler)
X_scaled_eval = scaler.transform(X)

# Must match training split
X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
    X_scaled_eval,
    y,
    test_size=0.2,
    random_state=42
)

# Predict only on test data
y_test_pred = model.predict(X_test_eval)


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/suggestions")
def suggestions():
    return render_template("suggestion.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print("Incoming Data:", data)

    gender = 1 if data["gender"] == "Male" else 0

    input_df = pd.DataFrame([{
        "age": float(data["age"]),
        "gender": gender,
        "subscription_length": float(data["subscription_length"]),
        "membership_type": float(data["membership_type"]),
        "num_logins": float(data["num_logins"]),
        "num_complaints": float(data["num_complaints"]),
        "num_classes_attended": float(data["num_classes_attended"]),
        "avg_session_time": float(data["avg_session_time"])
    }])

    print("Input DF:\n", input_df)

    scaled = scaler.transform(input_df)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    result = "Likely to Renew" if prediction == 1 else "Likely to Churn"

    return jsonify({
        "prediction": result,
        "probability": round(probability * 100, 2)
    })


# =========================
# API ENDPOINTS
# =========================

@app.route("/api/kpis")
def get_kpis():
    at_risk_count = df[df["risk_level"] == "High"].shape[0]
    churn_probability = df["predicted_probability"].mean() * 100
    churn_rate = (df["renewed_membership"] == 0).mean() * 100
    avg_visits = df["num_logins"].mean()

    return jsonify({
        "at_risk_count": int(at_risk_count),
        "churn_probability": round(churn_probability, 2),
        "churn_rate": round(churn_rate, 2),
        "avg_visits": round(avg_visits, 2)
    })

@app.route("/api/risk-distribution")
def risk_distribution():
    distribution = df["risk_level"].value_counts().to_dict()
    return jsonify({
        "low_risk": distribution.get("Low", 0),
        "medium_risk": distribution.get("Medium", 0),
        "high_risk": distribution.get("High", 0)
    })

@app.route("/api/churn-factors")
def churn_factors():
    importances = model.feature_importances_
    feature_names = model.feature_names_in_

    factors = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    return jsonify([
        {"feature": f[0], "importance": round(f[1], 3)}
        for f in factors
    ])

@app.route("/api/high-risk-members")
def high_risk_members():
    high_risk_df = df[df["risk_level"] == "High"]

    top_10 = high_risk_df.sort_values(
        by="predicted_probability",
        ascending=False
    ).head(10)

    result = top_10[[
        "customer_id",
        "num_logins",
        "num_complaints",
        "subscription_length",
        "risk_level",
        "suggested_action"
    ]].to_dict(orient="records")

    return jsonify(result)
@app.route("/api/suggestions-metrics")
def suggestions_metrics():
    total_members = len(df)
    actual_churn_rate = (df["renewed_membership"] == 0).mean() * 100
    at_risk = df[df["risk_level"] == "High"].shape[0]
    retention_rate = 100 - actual_churn_rate
    avg_lifetime = df["subscription_length"].mean()

    return jsonify({
        "total_members": total_members,
        "actual_churn_rate": round(actual_churn_rate, 2),
        "at_risk_members": at_risk,
        "retention_rate": round(retention_rate, 2),
        "avg_lifetime": round(avg_lifetime, 1)
    })

@app.route("/api/execution-roadmap")
def execution_roadmap():

    total_members = len(df)
    high_risk = df[df["risk_level"] == "High"].shape[0]
    complaint_avg = df["num_complaints"].mean()
    visit_avg = df["num_logins"].mean()

    high_risk_percent = (high_risk / total_members) * 100

    roadmap = []


    # Phase 1 Logic
    if complaint_avg > 2:
        roadmap.append({
            "phase": "Immediate (30 Days)",
            "title": "Complaint Resolution System",
            "reason": "High average complaint count detected",
            "action": "Launch 24hr complaint response workflow"
        })
    else:
        roadmap.append({
            "phase": "Immediate (30 Days)",
            "title": "Low Engagement Recovery",
            "reason": "Low average gym visits observed",
            "action": "Call top 50 low-visit members"
        })

    # Phase 2 Logic
    if visit_avg < 8:
        roadmap.append({
            "phase": "Short-Term (60 Days)",
            "title": "Engagement Campaign",
            "reason": "Members visiting less frequently",
            "action": "Launch class attendance challenge"
        })
    else:
        roadmap.append({
            "phase": "Short-Term (60 Days)",
            "title": "Membership Upsell Strategy",
            "reason": "Stable visits, focus revenue expansion",
            "action": "Promote 6/12 month memberships"
        })

    # Phase 3 Logic
    roadmap.append({
        "phase": "Long-Term (90+ Days)",
        "title": "Retention Culture System",
        "reason": f"{round(high_risk_percent,1)}% members classified high-risk",
        "action": "Build automated retention monitoring system"
    })

    return jsonify(roadmap)


@app.route("/api/business-impact")
def business_impact():

    total_members = len(df)

    current_churn_rate = (df["renewed_membership"] == 0).mean()

    high_risk_members = df[df["risk_level"] == "High"].shape[0]

    class_attendance_rate = (
        df["num_classes_attended"].sum() /
        df["subscription_length"].sum()
    ) * 100

    annual_membership_ratio = (
        (df["membership_type"] == 12).mean()
    ) * 100

    avg_complaints = df["num_complaints"].mean()

    avg_member_value = 1000  # keep static for now

    return jsonify({
        "total_members": total_members,
        "current_churn_rate": current_churn_rate,
        "high_risk_members": high_risk_members,
        "class_attendance_rate": class_attendance_rate,
        "annual_membership_ratio": annual_membership_ratio,
        "avg_complaints": avg_complaints,
        "avg_member_value": avg_member_value
    })


# =========================

if __name__ == "__main__":
    app.run(debug=True)


