import pandas as pd

def preprocess_input(data, scaler):

    # Encode gender properly
    gender = 1 if data["gender"] == "Male" else 0

    # Create DataFrame with EXACT feature names in EXACT order
    input_df = pd.DataFrame([{
        "age": float(data["age"]),
        "gender": float(gender),
        "subscription_length": float(data["subscription_length"]),
        "membership_type": float(data["membership_type"]),
        "num_logins": float(data["num_logins"]),
        "num_complaints": float(data["num_complaints"]),
        "num_classes_attended": float(data["num_classes_attended"]),
        "avg_session_time": float(data["avg_session_time"])
    }])

    scaled = scaler.transform(input_df)

    return scaled

