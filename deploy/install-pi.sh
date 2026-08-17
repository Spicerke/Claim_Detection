#!/usr/bin/env bash
#
# One-shot installer for the Claim Detection API on a Raspberry Pi.
# Run it FROM the repo checkout on the Pi:
#
#   ALLOWED_ORIGINS=https://spicerke.github.io ./deploy/install-pi.sh
#
# Assumes the model weights are already on disk (see deploy/README.md, step 2).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$USER}"
MODEL_DIR="${MODEL_DIR:-$HOME/claim-model}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-https://spicerke.github.io}"

echo "Repo:      $REPO"
echo "User:      $RUN_USER"
echo "Model dir: $MODEL_DIR"
echo "Origins:   $ALLOWED_ORIGINS"
echo

# --- 1. Sanity-check the weights before doing any work ---
for f in config.json model.safetensors tokenizer.json tokenizer_config.json; do
    if [[ ! -f "$MODEL_DIR/$f" ]]; then
        echo "ERROR: missing $MODEL_DIR/$f" >&2
        echo "Copy the model files to the Pi first (see deploy/README.md)." >&2
        exit 1
    fi
done
echo "✓ Model files present"

# --- 1b. Check free space before starting a 15-minute install ---
# The dependency tree lands at ~600-700MB installed, but pip needs roughly
# double that in flight (download + unpack). Fail now with a clear message
# rather than dying with ENOSPC partway through the torch install.
REQUIRED_MB=1500
AVAIL_MB="$(df -Pm "$REPO" | awk 'NR==2 {print $4}')"
if (( AVAIL_MB < REQUIRED_MB )); then
    echo "ERROR: only ${AVAIL_MB}MB free on the filesystem holding $REPO." >&2
    echo "Need ~${REQUIRED_MB}MB. Try:" >&2
    echo "  sudo raspi-config --expand-rootfs && sudo reboot   # if the card was never expanded" >&2
    echo "  sudo apt clean                                     # cached .debs" >&2
    exit 1
fi
echo "✓ ${AVAIL_MB}MB free"

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

echo "Installing dependencies (torch is large -- this takes a while on a Pi)..."
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
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo
        echo "✓ API is healthy at http://127.0.0.1:8000"
        curl -s http://127.0.0.1:8000/health
        echo
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo
echo "ERROR: API did not become healthy in 120s. Check: sudo journalctl -u claim-api -n 50" >&2
exit 1
