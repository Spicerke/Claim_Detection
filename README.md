# Project Overview 
This project is a full-stack machine learning application designed to identify whether a given sentence contains a fact checkable claim or assertion. The system classifies natural language input and returns a boolean value along with a confidence score.

## Objectives
- *Fine-tuning*: Train a lightweight encoder model (DistilBERT) for binary classification.
- *API Development*: Build a high-performance FastAPI backend.
- *Production Readiness*: Implement caching, security validation, and Docker containerization.
- *Testing*: Conduct load testing via Locust to simulate production traffic.

# System Architecture 
![image](Diagram.png)
## Breakdown 
- **Frontend**: 
    - *HTML/CSS*: Simple webpage allowing users to input text and view classification results (True/False + Confidence %).
    - *Flask*: In charge of routing data from the HTML webpage to the API and from the API back to the HTML page 
- **API Gateway** (FastAPI): Handles incoming POST requests. It includes:
    - *Input Validation*: Input string is length and IP validated to stop DDOS attacks 
    - _LRU Cache_: An in-memory cache to store common phrases, bypassing model inference for repeated queries to save CPU cycles.
- **Inference Engine**: A fine-tuned DistilBERT model. 
- **Deployment**: The entire stack is containerized using Docker for environment parity. 

# Data Modeling and Trainer 
*Datasets: [ClaimBuster](https://zenodo.org/records/3836810) & [FeverClaims](https://fever.ai/dataset/fever.html)*

## Data Proccessing 
In order to ensure that we had a good mix of claims and non-claims along with a breadth of domains 2 datasets were combined. The ClaimBusters dataset was our main dataset but it was weighted towards non-factual statments so the FeverClaims dataset was brought in and proccessed and combined with the Claimbusters to make our final training dataset 

**Mapping**: 
- Label 1: Check-worthy  statements.
-  Label 0: Non-Check-worthy statements.

## Training Workflow 
1. *Preprocessing*: Tokenization via DistilBertTokenizer.
2. *Split*: 80% training / 20% testing.
3. *Metrics*: Focus on F1-Score, Precision, and Recall.
    ```
    ==============================
    📊 MODEL EVALUATION METRICS 📊
    ==============================
    Accuracy:  0.9424
    F1 Score:  0.9416
    Precision: 0.9542
    Recall:    0.9294
    ==============================
    ```
    ![confusion matrix](FineTuning/confusion_matrix.png)
    ![Precison_curve](FineTuning/precision_recall_curve.png)

# API and Security 
**API: FastAPI* *
- **Endpoint**: ```POST /predict```
    - _Inputs_: ```{"text": "string"}```
    - _Outputs_: ```{"is_claim": boolean, "confidence": float}```
    - _Security_:  
        - Backend Validation: Enforced string length limits to prevent (DoS) via massive text payloads.
        - Rate Limiter: After load testing included a rate limiter to ensure the API can handle the load.

# Performance & Testing
*Load testing done via Locust*
- _Goal_: Determine the requests-per-second (RPS) threshold 
- _Result_: 83.95 RPS
    ![FailureGraph](Requests.png)

# Contanerization 
The application is wrapped in a multi-stage Docker build:

```
# To build 
docker-compose up --build
# To run if already build 
docker-compose up
```


# Setup 
## To run with pretrained model 
1. Clone the repo: ```git clone https://github.com/Spicerke/Claim_Detection```
2. Run ```docker-compose up --build ```
3. Go to: http://localhost:5001

## To train your own model 
1. Clone the repo: ```git clone https://github.com/Spicerke/Claim_Detection```
2. Install dependencies ``` pip install -r requirments.txt ```
3. Go to Finetuning folder. Can either use the preproccessed data or add your own data to the dataset folder
4. Point the DATA_FILE varible in Train.py to whatever data you want to use 
5. Run ```python Train.py```
6. To evaluate run ```python evaluate.py```
6. After training is complete to use app with new model run  ```docker-compose up --build ```
7. Go to: http://localhost:5001


## To run tests 
1. Clone the repo: ```git clone https://github.com/Spicerke/Claim_Detection```
2. Install dependencies ``` pip install -r requirments.txt ```
3. Go to the tests folder
4. run ```./runtests.sh```
5. Go to http://127.0.0.1:8089