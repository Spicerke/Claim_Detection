#!/usr/bin/env bash
#
# One-shot installer for the Claim Detection API on a Raspberry Pi.
# Run it FROM the repo checkout on the Pi:
#
#   ALLOWED_ORIGINS=https://spicerke.github.io ./deploy/install-pi.sh
#
# Set API_PORT if something else on the Pi already owns the default 8000:
#
#   API_PORT=8005 ALLOWED_ORIGINS=https://spicerke.github.io ./deploy/install-pi.sh
#
# Whatever port you pick must also be the one cloudflared forwards to.
#
# Assumes the model weights are already on disk (see deploy/README.md, step 2).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$USER}"
MODEL_DIR="${MODEL_DIR:-$HOME/claim-model}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-https://spicerke.github.io}"
API_PORT="${API_PORT:-8000}"

echo "Repo:      $REPO"
echo "User:      $RUN_USER"
echo "Model dir: $MODEL_DIR"
echo "Origins:   $ALLOWED_ORIGINS"
echo "Port:      $API_PORT"
echo

# --- 1. Sanity-check the model before doing any work ---
# Only two files are needed for inference: the exported ONNX graph and the
# tokenizer definition. Produce model.onnx with FineTuning/export_onnx.py.
for f in model.onnx tokenizer.json; do
    if [[ ! -f "$MODEL_DIR/$f" ]]; then
        echo "ERROR: missing $MODEL_DIR/$f" >&2
        echo "Copy the model files to the Pi first (see deploy/README.md)." >&2
        exit 1
    fi
done
# A truncated scp leaves a valid-looking but unloadable file; catch it here
# rather than at service start.
ONNX_MB=$(( $(stat -c %s "$MODEL_DIR/model.onnx" 2>/dev/null || echo 0) / 1048576 ))
if (( ONNX_MB < 200 )); then
    echo "ERROR: $MODEL_DIR/model.onnx is only ${ONNX_MB}MB, expected ~256MB." >&2
    echo "The transfer was probably truncated -- re-copy it." >&2
    exit 1
fi
echo "✓ Model files present (model.onnx ${ONNX_MB}MB)"

# --- 1b. Check free space before starting the install ---
# The dependency tree is ~150MB installed; pip needs roughly double in flight.
REQUIRED_MB=600
AVAIL_MB="$(df -Pm "$REPO" | awk 'NR==2 {print $4}')"
if (( AVAIL_MB < REQUIRED_MB )); then
    echo "ERROR: only ${AVAIL_MB}MB free on the filesystem holding $REPO." >&2
    echo "Need ~${REQUIRED_MB}MB. Try:" >&2
    echo "  sudo raspi-config --expand-rootfs && sudo reboot   # if the card was never expanded" >&2
    echo "  sudo apt clean                                     # cached .debs" >&2
    exit 1
fi
echo "✓ ${AVAIL_MB}MB free"

# --- 1d. Make sure nothing already owns the port ---
# uvicorn's bind failure surfaces as a bare "[Errno 98] address already in use"
# inside a restart loop, which reads like a model problem but isn't. Catch it
# here, while we can still name the process holding the port.
#
# Stop our own service first, or a re-run would trip over the copy of itself
# that the previous run started -- and its Restart=always would keep grabbing
# the port back between the check and the new unit taking over.
sudo systemctl stop claim-api 2>/dev/null || true
if command -v ss >/dev/null 2>&1 && ss -tln 2>/dev/null | grep -qE "[:.]${API_PORT}\b"; then
    echo "ERROR: something is already listening on port ${API_PORT}." >&2
    echo "Find it with:  sudo ss -tlnp | grep :${API_PORT}" >&2
    echo "Then stop it, or re-run with a different port:" >&2
    echo "  API_PORT=8005 ALLOWED_ORIGINS=$ALLOWED_ORIGINS ./deploy/install-pi.sh" >&2
    exit 1
fi
echo "✓ Port ${API_PORT} is free"

# --- 1c. Refuse to run on a stale venv that still has torch in it ---
# Installing over a torch venv leaves ~700MB of dead weight and, worse, an
# onnxruntime install that appears fine while the old SIGILL-ing torch is still
# importable. Start clean instead.
if [[ -d "$REPO/.venv" ]] && "$REPO/.venv/bin/python" -c "import torch" 2>/dev/null; then
    echo "Existing venv contains torch (no longer used). Removing it..."
    rm -rf "$REPO/.venv"
fi

# --- 2. Virtualenv ---
if [[ ! -d "$REPO/.venv" ]]; then
    echo "Creating virtualenv..."
    python3 -m venv "$REPO/.venv"
fi

# pip unpacks wheels under TMPDIR. On Pi images where /tmp is a tmpfs, unpacking
# torch can blow out a RAM-backed /tmp even with plenty of disk free, so point it
# at a disk-backed scratch dir we control.
PIP_TMP="$REPO/.piptmp"
mkdir -p "$PIP_TMP"
trap 'rm -rf "$PIP_TMP"' EXIT

echo "Installing dependencies (~150MB, a couple of minutes on a Pi)..."
# --no-cache-dir: without it pip keeps a second full copy of every wheel in
# ~/.cache/pip, which is several hundred MB of dead weight on an SD card.
TMPDIR="$PIP_TMP" "$REPO/.venv/bin/pip" install --no-cache-dir --upgrade pip
TMPDIR="$PIP_TMP" "$REPO/.venv/bin/pip" install --no-cache-dir -r "$REPO/App/requirements.txt"
echo "✓ Dependencies installed"

# --- 3. Render and install the systemd unit ---
TMP_UNIT="$(mktemp)"
sed -e "s|__USER__|$RUN_USER|g" \
    -e "s|__REPO__|$REPO|g" \
    -e "s|__MODEL_DIR__|$MODEL_DIR|g" \
    -e "s|__ORIGINS__|$ALLOWED_ORIGINS|g" \
    -e "s|__PORT__|$API_PORT|g" \
    "$REPO/deploy/claim-api.service" > "$TMP_UNIT"

sudo install -m 644 "$TMP_UNIT" /etc/systemd/system/claim-api.service
rm -f "$TMP_UNIT"

sudo systemctl daemon-reload
sudo systemctl enable claim-api
sudo systemctl restart claim-api
echo "✓ Service installed and started"

# --- 4. Wait for the model to load, then verify ---
echo -n "Waiting for the API to come up"
for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
        echo
        echo "✓ API is healthy at http://127.0.0.1:${API_PORT}"
        curl -s "http://127.0.0.1:${API_PORT}/health"
        echo
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo
echo "ERROR: API did not become healthy in 120s. Check: sudo journalctl -u claim-api -n 50" >&2
exit 1
