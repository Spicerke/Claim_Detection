from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware

# --- SlowAPI Imports ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

MODEL_DIR = "./claim_detection_model"

# Initialize Limiter using the client's IP address
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Claim Detection API",
    description="An API that determines if a natural language sentence contains a factual claim.",
    version="1.0.0"
)

# --- Register SlowAPI with FastAPI ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Caching System 
CACHE_MAX_SIZE = 10000 #local memory cache, would likely update to Redis or SQlite in next iterations 
prediction_cache = {}

print("Loading DistilBERT model into memory...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  

class ClaimRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, 
                      description="The natural language sentence to analyze.")

class ClaimResponse(BaseModel):
    is_claim: bool
    confidence: float
    cached: bool
    processing_time_ms: float

# API Endpoints

@app.get("/health")
@limiter.limit("10/minute") # Strict limit for health checks
def health_check(request: Request): # Request parameter is required by slowapi
    return {"status": "API is healthy and model is loaded."}

@app.post("/predict", response_model=ClaimResponse)
@limiter.limit("5/second") # Blocks spam, but allows standard UI usage
def predict_claim(request: Request, payload: ClaimRequest): # Renamed to payload to avoid collision
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

    # Tokenize 
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128, 
        padding="max_length"
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