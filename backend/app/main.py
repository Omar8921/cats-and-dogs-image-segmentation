from fastapi import FastAPI
from backend.app.api import health
from backend.app.api import segmentations
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from deep_learning import inference

WEIGHTS_PATH = 'deep_learning/model/model_weights.pth'

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    inference.init_inference(WEIGHTS_PATH)
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(segmentations.router)