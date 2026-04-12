#!/bin/bash

echo "Starting Claim Detection Application..."

# This function catches when you press Ctrl+C and cleanly kills both servers
cleanup() {
    kill $(jobs -p) 2>/dev/null
    wait $(jobs -p) 2>/dev/null
    echo "Done."
}

# Trap the Ctrl+C signal to trigger the cleanup function
trap cleanup SIGINT

# 1. Start the FastAPI Backend in the background (&)
echo "▶ Starting FastAPI backend on port 8000..."
uvicorn main:app --port 8000 &

# Give the API a quick 2 seconds to boot up before starting the frontend
sleep 2

# 2. Start the Flask Frontend in the background (&)
echo "▶ Starting Flask frontend on port 5000..."
cd frontend && python app.py &
echo "👉 Web App:      http://127.0.0.1:5000"
echo "👉 API Docs:     http://127.0.0.1:8000/docs"
echo "Press Ctrl+C to stop both servers."
echo ""
# Keep the script running to hold the background jobs open
wait