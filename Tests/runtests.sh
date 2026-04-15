#!/bin/bash

echo "🚀 Starting Test Environment..."

# Cleanup function to kill the app servers AND Locust
cleanup() {
    echo ""
    echo "🧹 Shutting down servers and test environment..."
    
    # 1. Kill standard background jobs
    kill $(jobs -p) 2>/dev/null
    
    # 2. Aggressively kill anything on the FastAPI (8000), Flask (5000), or Locust (8089) ports
    lsof -t -i:5000 -i:8000 -i:8089 | xargs kill -9 2>/dev/null
    
    echo "✅ Clean shutdown complete."
    exit 0
}

# Trap BOTH Ctrl+C (SIGINT) and script completion (EXIT)
trap cleanup SIGINT EXIT

# Navigate up to the root, then into the App folder
cd ../App || { echo "❌ App directory not found! Make sure you run this from inside the Tests folder."; exit 1; }

# 1. Start the FastAPI Backend
echo "▶ Starting FastAPI backend on port 8000..."
uvicorn main:app --port 8000 &

# Give the API 2 seconds to boot
sleep 2

# 2. Start the Flask Frontend
echo "▶ Starting Flask frontend on port 5000..."
cd frontend || { echo "❌ Frontend directory not found!"; exit 1; }
python app.py &

# Give Flask a second to boot
sleep 1

# Navigate back up to the root, then back into the Tests folder
cd ../../Tests || { echo "❌ Tests directory not found!"; exit 1; }

# 3. Start Locust Testing Environment
echo ""
echo "▶ Starting Locust load testing environment..."
echo "👉 Locust Web Interface: http://127.0.0.1:8089"
echo "👉 FastAPI Target:       http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the tests and shut down all servers."
echo ""

# Run Locust (since we are already in the Tests folder, we just point directly to locustfile.py)
locust -f locustfile.py --host=http://127.0.0.1:8000