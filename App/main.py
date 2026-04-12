from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware

MODEL_DIR = "/Users/kaispicer/Desktop/Claim_Detection/FineTuning/claim_detection_model"

app = FastAPI(
    title="Claim Detection API",
    description="An API that determines if a natural language sentence contains a factual claim.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

# Caching System 
CACHE_MAX_SIZE = 10000 #local memory cache, would likely update to Redis or SQlite in 
#next itetations 
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

#API Endpoints
@app.get("/health")
def health_check():
    return {"status": "API is healthy and model is loaded."}

@app.post("/predict", response_model=ClaimResponse)
def predict_claim(request: ClaimRequest):
    start_time = time.time()
    input_text = request.text.strip()

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
    
    # If the probability is > 70%, we classify it as a claim
    is_claim = bool(confidence_score_decimal > 0.7)
    confidence_pct = round(confidence_score_decimal * 100, 2)

    if len(prediction_cache) >= CACHE_MAX_SIZE:
        prediction_cache.pop(next(iter(prediction_cache)))
        
    prediction_cache[input_text] = {
        "is_claim": is_claim,
        "confidence": confidence_pct
    }
    process_time = (time.time() - start_time) * 1000

    #Return Response
    return ClaimResponse(
        is_claim=is_claim,
        confidence=confidence_pct,
        cached=False,
        processing_time_ms=round(process_time, 2)
    )