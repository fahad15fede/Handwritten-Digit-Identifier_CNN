# Handwritten Digit Identifier — CNN

A FastAPI backend that accepts a handwritten digit image and returns the predicted digit (0–9) along with confidence score and class probabilities, powered by a trained CNN model on the MNIST dataset.

---

## Project Structure

```
├── backend/
│   ├── app.py               # FastAPI app entry point
│   ├── routes.py            # API route definitions
│   ├── predictor.py         # Loads model, runs inference
│   ├── preprocess.py        # Full image preprocessing pipeline
│   ├── testPreprocessor.py  # Script to test preprocessing locally
│   ├── requirements.txt     # Python dependencies
│   ├── model/
│   │   └── mnist_final.keras  # Trained CNN model
│   ├── test_images/         # Sample digit images for testing
│   └── uploads/             # Uploaded images (runtime)
├── requirements.txt         # Pinned full dependency list
└── .gitignore
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Server

Navigate into the `backend` folder and start the server with uvicorn:

```bash
cd backend
uvicorn app:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

---

## API Endpoints

### `GET /`
Health check — confirms the API is running.

**Response:**
```json
{ "message": "API Running" }
```

### `GET /health`
Checks if the model loaded successfully.

**Response:**
```json
{ "status": "healthy", "model_loaded": true }
```

### `POST /predict` *(to be wired up in routes.py)*
Upload a handwritten digit image and receive the prediction.

**Request:** `multipart/form-data` with an image file field.

**Response:**
```json
{
  "digit": 7,
  "confidence": 98.43,
  "probabilities": [0.001, 0.002, ..., 0.984, ...]
}
```

---

## Testing the Preprocessor

To visually verify each preprocessing step using a sample image, run:

```bash
cd backend
python testPreprocessor.py
```

This saves step-by-step output images to `backend/output/`:
- `original.png` — raw decoded image
- `gray.png` — grayscale
- `blur.png` — Gaussian blurred
- `threshold.png` — binary thresholded

> Make sure the `backend/output/` directory exists before running.

---

## Code Walkthrough

### `preprocess.py` — Image Pipeline

The preprocessing pipeline transforms a raw uploaded image into the exact format the CNN expects `(1, 28, 28, 1)`:

| Step | Function | What it does |
|---|---|---|
| 1 | `read_image` | Decodes raw bytes into an OpenCV BGR image |
| 2 | `to_grayscale` | Converts BGR → grayscale |
| 3 | `blur_image` | Applies 5×5 Gaussian blur to reduce noise |
| 4 | `threshold_image` | Otsu's binarization — inverts so digit is white on black (matches MNIST format) |
| 5 | `extract_digit` | Finds the largest contour and crops tightly around the digit |
| 6 | `resize_keep_aspect` | Scales digit to fit within 20×20, preserving aspect ratio |
| 7 | `pad_image` | Centers the 20×20 digit in a 28×28 canvas with zero padding |
| 8 | `normalize` | Scales pixel values from `[0, 255]` to `[0.0, 1.0]` |
| 9 | `reshape_image` | Reshapes to `(1, 28, 28, 1)` — batch size 1, single channel |

The resize + pad approach (steps 6–7) is intentional — it replicates how MNIST digits are formatted, which improves model accuracy on real-world input.

---

### `predictor.py` — Model Inference

```python
model = tf.keras.models.load_model('model/mnist_final.keras')
```

The model is loaded once at startup (module level), so it's not reloaded on every request.

The `predict(image)` function:
- Runs `model.predict()` on the preprocessed `(1, 28, 28, 1)` array
- Uses `np.argmax` to get the predicted digit class (0–9)
- Returns digit, confidence percentage, and full probability distribution across all 10 classes

---

### `app.py` — FastAPI App

Minimal entry point. Creates the FastAPI instance and registers the router from `routes.py`.

---

### `routes.py` — API Routes

Defines the API endpoints and imports the loaded model from `predictor.py`. The model import at the top of the file triggers loading at server startup, not per-request.

---

## Model

- Architecture: Convolutional Neural Network (CNN)
- Dataset: MNIST (70,000 handwritten digit images)
- Input shape: `(28, 28, 1)` — grayscale, normalized
- Output: Softmax over 10 classes (digits 0–9)
- Format: Keras `.keras` native format
