import os
import time

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- SlowAPI Imports ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Configuration (env-driven so the Pi and a laptop can differ) ---

# On the Pi the weights live outside the repo and are bind-mounted / pointed at
# via this variable. Locally it falls back to the folder next to this file.
MODEL_DIR = os.getenv("MODEL_DIR", "./claim_detection_model")

# Browsers block cross-origin calls unless the API says the origin is allowed.
# The GitHub Pages frontend is a *different* origin from the tunnel, so its URL
# must be listed here. Comma-separated, no trailing slash.
ALLOWED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
]

# A Pi has few cores. Letting torch grab all of them for one request starves
# concurrent requests and thrashes. 2 is a sane default for a 4-core Pi.
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))


def get_client_ip(request: Request) -> str:
    """
    Rate-limit key that survives the Cloudflare Tunnel.

    cloudflared connects to uvicorn over loopback, so every request arrives with
    a remote address of 127.0.0.1 -- without this, all users on earth would share
    a single rate-limit bucket. Cloudflare sets CF-Connecting-IP to the real
    client IP.

    This header is only trustworthy because uvicorn binds to 127.0.0.1, so
    cloudflared is the only thing that can reach it. If you ever expose the port
    directly, a client could forge this header to dodge the limiter.
    """
    return request.headers.get("cf-connecting-ip") or get_remote_address(request)


limiter = Limiter(key_func=get_client_ip)

app = FastAPI(
    title="Claim Detection API",
    description="An API that determines if a natural language sentence contains a factual claim.",
    version="1.0.0",
)

# --- Register SlowAPI with FastAPI ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS: lets the static GitHub Pages frontend call this API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=86400,  # cache the preflight for a day
)

# Caching System
CACHE_MAX_SIZE = 10000  # local memory cache, would likely update to Redis or SQlite in next iterations
prediction_cache = {}

print(f"Loading DistilBERT model into memory from {MODEL_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
print("Model loaded.")


class ClaimRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000,
                      description="The natural language sentence to analyze.")


class ClaimResponse(BaseModel):
    is_claim: bool
    confidence: float
    cached: bool
    processing_time_ms: float


# API Endpoints

@app.get("/")
def root():
    """Friendly landing response so hitting the tunnel root isn't a 404."""
    return {
        "service": "Claim Detection API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
@limiter.limit("60/minute")  # generous: uptime monitors and the frontend warm-up both poll this
def health_check(request: Request):  # Request parameter is required by slowapi
    return {"status": "API is healthy and model is loaded."}


@app.post("/predict", response_model=ClaimResponse)
@limiter.limit("5/second")  # Blocks spam, but allows standard UI usage
def predict_claim(request: Request, payload: ClaimRequest):  # Renamed to payload to avoid collision
    start_time = time.time()
    input_text = payload.text.strip()

    if input_text in prediction_cache:
        cached_result = prediction_cache[input_text]
        process_time = (time.time() - start_time) * 1000
        return ClaimResponse(
            is_claim=cached_result["is_claim"],
            confidence=cached_result["confidence"],
            cached=True,
            processing_time_ms=round(process_time, 2)
        )

    # Tokenize. No padding: this is a single sequence, so padding to a fixed 128
    # tokens just makes the Pi run attention over tokens that are masked out anyway.
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    # Model Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence_score_decimal = probs[0][1].item()

    # If the probability is > 50%, we classify it as a claim
    is_claim = bool(confidence_score_decimal > 0.5)
    confidence_pct = round(confidence_score_decimal * 100, 2)

    if is_claim == False:
        confidence_pct = round((1 - confidence_score_decimal) * 100, 2)

    if len(prediction_cache) >= CACHE_MAX_SIZE:
        prediction_cache.pop(next(iter(prediction_cache)))

    prediction_cache[input_text] = {
        "is_claim": is_claim,
        "confidence": confidence_pct
    }
    process_time = (time.time() - start_time) * 1000

    # Return Response
    return ClaimResponse(
        is_claim=is_claim,
        confidence=confidence_pct,
        cached=False,
        processing_time_ms=round(process_time, 2)
    )
