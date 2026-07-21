from fastapi import APIRouter, UploadFile, File

from preprocess import preprocess
from predictor import predict_digits

router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "Handwritten Digit Recognition API Running"
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_bytes = await file.read()

    processed_digits = preprocess(file_bytes)

    result = predict_digits(processed_digits)

    return result