from preprocessing import preprocess_input

def predict_churn(data, model, scaler):
    processed_data = preprocess_input(data, scaler)

    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    return int(prediction), float(probability)
