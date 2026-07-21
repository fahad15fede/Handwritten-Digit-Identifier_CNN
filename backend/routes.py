from fastapi import APIRouter
from predictor import model

router = APIRouter()


@router.get("/")
def home():
    return {"message": "API Running"}


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }