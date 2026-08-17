# Production Deployment

Two halves, deployed independently:

| Piece | Runs on | URL |
|---|---|---|
| FastAPI + DistilBERT weights | Raspberry Pi, `systemd`, bound to `127.0.0.1:8000` | `https://claims.yourdomain.com` (via Cloudflare Tunnel) |
| Static frontend | GitHub Pages, auto-deployed by Actions | `https://spicerke.github.io/Claim_Detection/` |

The browser calls the tunnel hostname directly. There is no server-side proxy —
which is why CORS and the rate-limiter's IP handling both matter.

```
Browser ──▶ spicerke.github.io (static HTML/JS)
   │
   └── fetch() ──▶ Cloudflare edge ──▶ tunnel ──▶ cloudflared on Pi ──▶ 127.0.0.1:8000
```

---

## Part 1 — Backend on the Raspberry Pi

### Prerequisites

**A 64-bit OS is required.** PyTorch publishes no wheels for 32-bit ARM, so
`pip install torch` fails on a 32-bit Pi OS. Confirm:

```bash
uname -m          # must print aarch64, not armv7l
free -h           # 2GB is tight; the process sits around 700MB-1GB RSS
df -h /           # need ~2GB free: ~700MB deps + 270MB model + install headroom
df -h /tmp        # if this says tmpfs, see the ENOSPC note below
```

If `uname -m` prints `armv7l`, reflash with the 64-bit Raspberry Pi OS image.

### Disk footprint

| | Size |
|---|---|
| `onnxruntime` | ~16 MB |
| `numpy` | ~16 MB |
| `tokenizers` | ~9 MB |
| `fastapi` + `uvicorn` + `pydantic` + `slowapi` | ~15 MB |
| **venv total** | **~150 MB** |
| `model.onnx` | 256 MB |

`install-pi.sh` checks for 600 MB free up front, passes `--no-cache-dir` (pip
otherwise keeps a second copy of every wheel in `~/.cache/pip`), and points
`TMPDIR` at a disk-backed scratch dir — Pi OS mounts `/tmp` as a RAM-backed
tmpfs, which can overflow during unpacking even with plenty of disk free.

## Why ONNX Runtime and not PyTorch

This is the part worth knowing if you ever revisit the dependency list.

**`pip install torch` does not work on a Raspberry Pi 5.** It installs, then dies
with `Fatal Python error: Illegal instruction` inside `torch/__init__.py` the
moment `libtorch_cpu.so` loads. The aarch64 torch wheels are built with SIMD
kernels targeting Neoverse-class server cores; the Pi 5's Cortex-A76 implements
ARMv8.2-A but has neither SVE nor BF16, so those kernels are literally
un-executable instructions. The wheel size tells the story — 99 MB at torch
2.9.1, 139 MB at 2.10.0, 400 MB at 2.11.0.

There is also a second, independent torch problem on ARM: from **2.11.0** torch
stopped gating its CUDA dependencies on `platform_machine == "x86_64"` (the
marker is now just `platform_system == "Linux"`). NVIDIA publishes aarch64
wheels, so pip cheerfully installs ~1.25 GB of `nvidia-cudnn` / `nccl` /
`cusparselt` / `triton` — about 2.5 GB unpacked — onto a board with no GPU. That
overflows the tmpfs `/tmp` and fails with `[Errno 28] No space left on device`
while `df -h /` still shows tens of GB free.

ONNX Runtime sidesteps both: baseline ARMv8 builds, no CUDA dependencies, 16 MB,
and **measured 3.4x faster** than torch on this model. The tradeoff is the export
step, and that inference code can no longer use `transformers` — but
`tokenizer.json` is a self-contained fast-tokenizer definition, so the `tokenizers`
package alone reproduces the exact same token ids.

Verified before the switch: identical token ids and identical predictions across
12 adversarial inputs (empty-ish, unicode, emoji, 200-word truncation, mixed
case, punctuation, HTML), with a max logit delta of `1.19e-06` between torch and
ONNX.

### `[Errno 28] No space left on device`

Should not happen now that torch is gone (~150 MB total), but if it does:

1. **The filesystem was never expanded.** If you flashed a 32GB card but `df -h /`
   shows only a few GB: `sudo raspi-config --expand-rootfs && sudo reboot`
2. **Genuinely full.** `sudo apt clean`, then
   `du -sh /home/* /var/log | sort -rh | head`.

### 1. Clone the repo on the Pi

```bash
sudo apt update && sudo apt install -y python3-venv git
git clone https://github.com/Spicerke/Claim_Detection.git ~/Claim_Detection
```

### 2. Export the model to ONNX and copy it over

The Pi runs inference on ONNX Runtime, not PyTorch. Export first, on the machine
that has torch installed:

```bash
# On your Mac, from the repo root:
python FineTuning/export_onnx.py
```

This writes `App/claim_detection_model/model.onnx` (~256MB, weights inlined) and
verifies it against the torch model before declaring success — it exits non-zero
if the outputs diverge.

Then copy the only two files the API needs:

```bash
ssh pi@raspberrypi.local 'mkdir -p ~/claim-model'
scp App/claim_detection_model/{model.onnx,tokenizer.json} \
    pi@raspberrypi.local:~/claim-model/
```

Nothing else goes over: not `model.safetensors`, not `config.json`, and
certainly not the `checkpoint-*` folders (2.3GB of optimizer and RNG state used
only for resuming training).

### 3. Install and start the service

```bash
# On the Pi:
cd ~/Claim_Detection
ALLOWED_ORIGINS=https://spicerke.github.io ./deploy/install-pi.sh
```

This creates a venv, installs dependencies, renders
[`claim-api.service`](claim-api.service) into `/etc/systemd/system/`, enables it
at boot, and waits for `/health` to answer.

**`ALLOWED_ORIGINS` must be the origin only** — scheme + host, no path, no
trailing slash. For `https://spicerke.github.io/Claim_Detection/`, the origin is
`https://spicerke.github.io`.

Verify locally on the Pi:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"The Eiffel Tower is 330 meters tall."}'
```

### 4. Point the tunnel at it

Add an ingress rule to your existing `cloudflared` config (see
[`cloudflared-config.example.yml`](cloudflared-config.example.yml)):

```yaml
ingress:
  - hostname: claims.yourdomain.com
    service: http://localhost:8000
    originRequest:
      connectTimeout: 30s
  - service: http_status:404
```

Then create the DNS record and restart:

```bash
cloudflared tunnel route dns <TUNNEL-NAME> claims.yourdomain.com
sudo systemctl restart cloudflared
```

Verify from your Mac:

```bash
curl https://claims.yourdomain.com/health
```

### Operating it

```bash
sudo systemctl status claim-api         # is it up
sudo journalctl -u claim-api -f         # live logs
sudo systemctl restart claim-api        # after a git pull or model swap
```

To ship a code change: `git pull` on the Pi, then `sudo systemctl restart claim-api`.
To swap models: `scp` new files into `~/claim-model/`, then restart.

---

## Part 2 — Frontend on GitHub Pages

### 1. Set your API hostname

Edit [`App/Frontend/config.js`](../App/Frontend/config.js):

```js
window.CLAIM_API_BASE = "https://claims.yourdomain.com";
```

This is a static site, so the URL is baked in at commit time — there is no build
step or environment variable to inject it.

### 2. Enable Pages

GitHub repo → **Settings** → **Pages** → **Source: GitHub Actions**.

Do *not* pick "Deploy from a branch" — the included workflow
([`.github/workflows/pages.yml`](../.github/workflows/pages.yml)) uses the
Actions source and will not run otherwise.

### 3. Push

```bash
git add -A && git commit -m "Productionize: static frontend + Pi backend" && git push
```

The workflow publishes `App/Frontend/` to
`https://spicerke.github.io/Claim_Detection/`. It reruns on any push touching
that folder.

---

## Why the code changed

Three things in the original app were incompatible with this split:

**The frontend was Flask.** GitHub Pages serves static files only — it cannot run
Python. `App/Frontend/` is now `index.html` + `app.js` + `config.js`, calling the
API with `fetch()` from the browser instead of proxying through a server.

**CORS was imported but never registered.** `main.py` imported `CORSMiddleware`
and never called `add_middleware`. With the frontend on a different origin than
the API, every browser request would have failed preflight. It is now registered
with an env-driven allowlist.

**Rate limiting counted the wrong IP.** `slowapi`'s `get_remote_address` reads
the socket peer, which behind `cloudflared` is always `127.0.0.1` — so the
`5/second` limit would have been a single global bucket shared by every user on
the internet. It now reads `CF-Connecting-IP`, which is safe *specifically
because* uvicorn binds to loopback: only `cloudflared` can reach it, so nobody
can forge that header. **If you ever change `--host 127.0.0.1` to `0.0.0.0`,
that guarantee breaks** and clients can spoof the header to bypass the limiter.

---

## Troubleshooting

**Browser console: "blocked by CORS policy"** — the origin isn't allowlisted.
Check `sudo systemctl show claim-api -p Environment` and confirm
`ALLOWED_ORIGINS` contains exactly `https://spicerke.github.io`. Restart after
editing the unit (`sudo systemctl daemon-reload && sudo systemctl restart claim-api`).

**502 from the tunnel** — `cloudflared` is up but the API isn't.
`sudo systemctl status claim-api`.

**Service won't start, logs show `NoSuchFile` / `Failed to load model`** —
`MODEL_DIR` is wrong or the files didn't copy. `ls -la ~/claim-model` should show
exactly two files: `model.onnx` at ~256MB and `tokenizer.json` at ~700KB.

**`Fatal Python error: Illegal instruction`** — something reinstalled torch.
`.venv/bin/pip list | grep -i torch` should return nothing; see "Why ONNX Runtime
and not PyTorch" above.

**Service killed during startup** — out of memory. Check `dmesg | grep -i oom`.
Add swap or use a Pi with more RAM.

**429s during normal use** — the `/predict` limit is `5/second` per IP. Adjust
the `@limiter.limit` decorator in `App/main.py` if that's too tight.
