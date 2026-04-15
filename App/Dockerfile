FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (caches the heavy PyTorch download)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code and the ML model folder
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]