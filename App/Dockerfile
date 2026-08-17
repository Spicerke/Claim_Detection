FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (caches the heavy PyTorch download)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application code. The model weights are ~270MB (and the training
# checkpoints 2.3GB); they are mounted at runtime via MODEL_DIR instead of being
# baked into the image.
COPY main.py .

ENV MODEL_DIR=/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
