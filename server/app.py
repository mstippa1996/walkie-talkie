"""Small YAMNet audio-classification service.

POST /classify   body: raw little-endian float32 mono PCM at 16 kHz, values in [-1, 1]
                 returns singing / speech scores plus the top AudioSet classes
GET  /health     reports whether the model has loaded
"""
import csv
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# Where to load YAMNet from: a TF Hub / Kaggle handle, or a local SavedModel directory.
MODEL_HANDLE = os.environ.get("YAMNET_MODEL", "https://tfhub.dev/google/yamnet/1")
# Comma-separated browser origins allowed to call this service.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
# If set, requests must send a matching "x-api-key" header.
API_KEY = os.environ.get("API_KEY")

SAMPLE_RATE = 16000
MIN_SAMPLES = 15600  # YAMNet needs at least one 0.975 s frame
MAX_SAMPLES = SAMPLE_RATE * 10  # reject anything longer than 10 s

SINGING_NAMES = {"Singing", "Choir", "Child singing", "Synthetic singing"}
SPEECH_NAMES = {
    "Speech",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Babbling",
}

state = {}


def load_class_names(model):
    path = model.class_map_path().numpy().decode("utf-8")
    with tf.io.gfile.GFile(path) as f:
        return [row["display_name"] for row in csv.DictReader(io.StringIO(f.read()))]


@asynccontextmanager
async def lifespan(app):
    model = hub.load(MODEL_HANDLE)
    names = load_class_names(model)
    state["model"] = model
    state["names"] = names
    state["singing_idx"] = [i for i, n in enumerate(names) if n in SINGING_NAMES]
    state["speech_idx"] = [i for i, n in enumerate(names) if n in SPEECH_NAMES]
    yield
    state.clear()


app = FastAPI(title="YAMNet classifier", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["content-type", "x-api-key"],
)


def run_model(waveform: np.ndarray):
    scores, _embeddings, _spectrogram = state["model"](waveform)
    return scores.numpy().mean(axis=0)  # average over the 0.96 s frames -> (521,)


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "loading"}


@app.post("/classify")
async def classify(request: Request):
    if API_KEY and request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    body = await request.body()
    if len(body) % 4 != 0:
        raise HTTPException(status_code=400, detail="Body must be float32 samples (length multiple of 4)")

    waveform = np.frombuffer(body, dtype="<f4")
    if waveform.size < MIN_SAMPLES:
        raise HTTPException(status_code=400, detail=f"Need at least {MIN_SAMPLES} samples (~1 s at 16 kHz)")
    if waveform.size > MAX_SAMPLES:
        raise HTTPException(status_code=413, detail="Audio too long (max 10 s)")
    if not np.isfinite(waveform).all():
        raise HTTPException(status_code=400, detail="Samples contain NaN or Inf")

    waveform = np.clip(waveform, -1.0, 1.0)
    mean_scores = await run_in_threadpool(run_model, waveform)

    names = state["names"]
    top = np.argsort(mean_scores)[::-1][:5]
    return {
        "singing": float(sum(mean_scores[i] for i in state["singing_idx"])),
        "speech": float(sum(mean_scores[i] for i in state["speech_idx"])),
        "top": [{"label": names[i], "score": float(mean_scores[i])} for i in top],
    }
