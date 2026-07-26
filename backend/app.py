from fastapi import FastAPI
<<<<<<< HEAD
from fastapi.middleware.cors import CORSMiddleware

=======
>>>>>>> bd404719447a51b21910d1d9b45ac1616c4ec352
from routes import router

app = FastAPI(
    title="Handwritten Digit Recognition API",
    version="1.0.0"
)

<<<<<<< HEAD
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

=======
>>>>>>> bd404719447a51b21910d1d9b45ac1616c4ec352
app.include_router(router)