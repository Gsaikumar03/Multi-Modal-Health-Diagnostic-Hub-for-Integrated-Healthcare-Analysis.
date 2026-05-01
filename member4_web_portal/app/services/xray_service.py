import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("densenet_pneumonia.h5", compile=False)

IMG_SIZE = 224


def predict_xray(file):

    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    return {
        "pneumonia_probability": float(prediction),
        "prediction": int(prediction > 0.5)
    }