import tensorflow as tf
import numpy as np

MODEL_PATH = "model/mnist_final.keras"

print("Loading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")


def predict(image):

    prediction = model.predict(image, verbose=0)

    digit = int(np.argmax(prediction))

    confidence = float(np.max(prediction) * 100)

    probabilities = prediction[0].tolist()

    return digit, confidence, probabilities


def predict_digits(images):

    digits = []

    confidences = []

    probabilities = []

    for image in images:

        digit, confidence, probs = predict(image)

        digits.append(str(digit))

        confidences.append(round(confidence, 2))

        probabilities.append(probs)

    phone_number = "".join(digits)

    return {
        "phone_number": phone_number,
        "digits": digits,
        "confidence": confidences,
        "probabilities": probabilities
    }