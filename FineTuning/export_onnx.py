"""
Export the fine-tuned DistilBERT classifier to ONNX for CPU inference.

Run this on your training machine (it needs torch + transformers). The Pi only
needs the two files it produces -- model.onnx and tokenizer.json -- and never
installs torch at all.

    python FineTuning/export_onnx.py

Why ONNX on the Pi: the aarch64 torch wheels are built with SIMD kernels that
assume Neoverse-class cores. A Raspberry Pi 5 (Cortex-A76) has no SVE and no
BF16, so libtorch_cpu.so dies with SIGILL on import. onnxruntime ships baseline
ARMv8 builds, is ~50MB instead of ~700MB, and is faster on CPU for a model this
size.
"""

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.getenv("MODEL_DIR", "App/claim_detection_model")
OUTPUT = os.path.join(MODEL_DIR, "model.onnx")
MAX_LENGTH = 128

# Opset 18. Asking for anything lower is pointless: torch's exporter emits a
# fused LayerNormalization node, and onnx's version converter cannot lower it
# ("No Previous Version of LayerNormalization exists"), so the downgrade fails
# silently and you get 18 anyway. Requires onnxruntime >= 1.17.
OPSET = 18


def main():
    print(f"Loading model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # Trace with a real sentence. Length doesn't matter -- the dynamic axes below
    # let the graph accept any batch size and sequence length at runtime.
    sample = tokenizer(
        "The Eiffel Tower is 330 meters tall.",
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    print(f"Exporting to {OUTPUT} (opset {OPSET}) ...")
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        OUTPUT,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )

    consolidate()
    verify(tokenizer, model)


def consolidate():
    """
    Fold the weights back into a single .onnx file.

    torch's exporter writes initializers to a sidecar `model.onnx.data`, leaving
    the .onnx itself under 1MB. That's two files to keep in sync on the Pi and an
    easy way to ship a model that loads but has no weights. DistilBERT is ~256MB,
    well under protobuf's 2GB ceiling, so there's no reason not to inline it.
    """
    import onnx

    sidecar = OUTPUT + ".data"
    if not os.path.exists(sidecar):
        return

    print("Inlining external weights into a single file ...")
    model = onnx.load(OUTPUT)  # pulls the sidecar into memory
    onnx.save_model(model, OUTPUT, save_as_external_data=False)
    os.remove(sidecar)


def verify(tokenizer, model):
    """Confirm the ONNX graph reproduces the torch model before we ship it."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(OUTPUT))
    session = ort.InferenceSession(OUTPUT, providers=["CPUExecutionProvider"])

    # Deliberately varied lengths -- this is what would catch a bad dynamic axis.
    cases = [
        "The Eiffel Tower is 330 meters tall.",
        "I think chocolate ice cream is the best.",
        "Unemployment fell to 3.5 percent in 2019 according to the BLS.",
        "Wow!",
        "The " + "very " * 100 + "long sentence about nothing in particular.",
    ]

    print("\nVerifying ONNX output against torch:")
    worst = 0.0
    for text in cases:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

        with torch.no_grad():
            torch_logits = model(**enc).logits.numpy()

        onnx_logits = session.run(
            None,
            {
                "input_ids": enc["input_ids"].numpy().astype(np.int64),
                "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
            },
        )[0]

        delta = float(np.abs(torch_logits - onnx_logits).max())
        worst = max(worst, delta)
        p_torch = float(torch.softmax(torch.tensor(torch_logits), -1)[0][1])
        p_onnx = float(torch.softmax(torch.tensor(onnx_logits), -1)[0][1])
        print(f"  len={enc['input_ids'].shape[1]:3}  torch={p_torch:.8f}  onnx={p_onnx:.8f}  delta={delta:.2e}")

    print(f"\nMax logit delta: {worst:.2e}")
    if worst > 1e-4:
        raise SystemExit(f"ERROR: ONNX output diverges from torch (max delta {worst:.2e})")

    size_mb = os.path.getsize(OUTPUT) / 1048576
    print(f"✓ Export verified. {OUTPUT} is {size_mb:.0f} MB")
    print("\nCopy these two files to the Pi's model directory:")
    print(f"  {OUTPUT}")
    print(f"  {os.path.join(MODEL_DIR, 'tokenizer.json')}")


if __name__ == "__main__":
    main()
