import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

# load model
model = tf.keras.models.load_model("heart_model.h5")

# load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# load encoder (feature columns)
with open("encoder.pkl", "rb") as f:
    feature_names = pickle.load(f)


def predict_heart(input_data: dict):

    # convert input to dataframe
    df = pd.DataFrame([input_data])

    # apply same encoding as training
    df = pd.get_dummies(df)

    # match training columns
    df = df.reindex(columns=feature_names, fill_value=0)

    # scaling
    input_scaled = scaler.transform(df)

    # prediction
    prediction = model.predict(input_scaled)[0][0]

    return {
        "risk_probability": float(prediction),
        "prediction": int(prediction > 0.5)
    }