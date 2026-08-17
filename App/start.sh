#!/bin/bash
#
# Local development. Production uses systemd on the Pi + GitHub Pages --
# see ../deploy/README.md.
#
# Serves the static frontend on :5500 and the API on :8000. Point
# Frontend/config.js at http://localhost:8000 while developing.

echo "Starting Claim Detection Application..."

cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $(jobs -p) 2>/dev/null
    lsof -t -i:5500 -i:8000 | xargs kill -9 2>/dev/null
    echo "Done."
    exit 0
}

trap cleanup SIGINT

cd "$(dirname "$0")"

# The static frontend runs on a different origin than the API, so the API has to
# allow it explicitly -- same mechanism as GitHub Pages in production.
export ALLOWED_ORIGINS="http://localhost:5500,http://127.0.0.1:5500"
export MODEL_DIR="${MODEL_DIR:-./claim_detection_model}"

echo "▶ Starting FastAPI backend on port 8000..."
uvicorn main:app --port 8000 --reload &

sleep 2

echo "▶ Serving static frontend on port 5500..."
(cd Frontend && python3 -m http.server 5500) &

echo ""
echo "👉 Web App:      http://127.0.0.1:5500"
echo "👉 API Docs:     http://127.0.0.1:8000/docs"
echo "Press Ctrl+C to stop both servers."
echo ""

wait
