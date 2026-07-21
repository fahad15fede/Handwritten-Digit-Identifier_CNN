import tensorflow as tf
import numpy as np

MODEL_PATH = 'model/mnist_final.keras'

print("Loading CNN model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

def predict(image):
    """
    image shape: (1, 28, 28, 1)
    """

    prediction = model.predict(image, verbose=0)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)*100)
    probabilities = prediction[0].tolist()

    return{
        "digit": digit,
        "confidence": round(confidence, 2),
        "probabilities": probabilities  
    }