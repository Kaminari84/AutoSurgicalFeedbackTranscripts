#!/usr/bin/env python3
"""classify_jama_cli.py

Binary per-sentence classification for the 6 JAMA feedback categories.

This script is intentionally separate from `classify_cli.py` (Irrelevant) so you
can change model types / inputs independently later.

Default expected model layout (local directories):

  classifiers/jama_classifiers/
    f_anatomic/threshold.json
    f_procedural/threshold.json
    f_technical/threshold.json
    f_visual_aid/threshold.json
    f_praise/threshold.json
    f_criticism/threshold.json

Each label directory should be loadable by Hugging Face:
  AutoTokenizer.from_pretrained(<dir>)
  AutoModelForSequenceClassification.from_pretrained(<dir>)

threshold.json can be either:
  - {"best_threshold": 0.52}
  - {"threshold": 0.52}
  - {"<LABEL>": 0.52}
  - 0.52

Outputs are appended to the input CSV:
  - <label>                     (0/1)
  - prob_<label>                (float)
  - conf_<label>                (float, rounded to 2 decimals)

Status JSON is written frequently so the Streamlit worker can surface progress.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Device selection (mirrors transcribe_cli style)
# -------------------------

def pick_device(requested: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Prefer CUDA by default, but only if it is actually usable."""
    req = (requested or "auto").strip().lower()

    import torch

    def cuda_probe() -> None:
        x = torch.randn((256, 256), device="cuda")
        y = x @ x
        _ = float(y.mean().item())

    if req not in {"auto", "cuda", "cpu"}:
        return "cpu", f"unknown --device={requested!r}; using cpu"

    if req == "cpu":
        return "cpu", None

    if torch.version.cuda is None:
        if req == "cuda":
            return "cpu", "CUDA requested but this torch build has torch.version.cuda=None (CPU-only build)"
        return "cpu", "torch is CPU-only (torch.version.cuda=None)"

    if not torch.cuda.is_available():
        if req == "cuda":
            return "cpu", "CUDA requested but torch.cuda.is_available() is False"
        return "cpu", "torch.cuda.is_available() is False"

    try:
        cuda_probe()
        return "cuda", None
    except Exception as e:
        if req == "cuda":
            return "cpu", f"CUDA requested but probe failed: {e}"
        return "cpu", f"CUDA probe failed; falling back to CPU: {e}"


# -------------------------
# JSON status utilities
# -------------------------

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


@dataclass
class StatusWriter:
    path: Path
    status: dict
    every_sec: float = 0.5
    _last: float = 0.0

    def update(self, *, force: bool = False, **fields: Any) -> None:
        now = time.time()
        for k, v in fields.items():
            self.status[k] = v
        self.status["updated"] = _now_iso()
        if force or (now - self._last) >= self.every_sec:
            atomic_write_json(self.path, self.status)
            self._last = now


# -------------------------
# Threshold + inference helpers
# -------------------------

def load_threshold(th_path: Path, label: str) -> float:
    if not th_path.exists():
        raise RuntimeError(f"threshold.json not found for {label!r}: {th_path}")

    obj = json.loads(th_path.read_text())

    if isinstance(obj, (float, int)):
        return float(obj)

    if isinstance(obj, dict):
        for k in ("best_threshold", "threshold", label):
            if k in obj:
                try:
                    return float(obj[k])
                except Exception:
                    pass

        # Some training scripts save nested structures; try the first numeric value.
        for v in obj.values():
            if isinstance(v, (float, int)):
                return float(v)

    raise RuntimeError(f"Could not parse threshold for {label!r} from {th_path}")


def predict_probs(
    texts: List[str],
    tokenizer,
    model,
    device,
    *,
    batch_size: int = 32,
    max_length: int = 256,
    sw: Optional[StatusWriter] = None,
    label: str = "",
    base_progress: int = 0,
    span_progress: int = 0,
) -> np.ndarray:
    """Return positive-class probabilities for each text."""
    import torch

    n = len(texts)
    probs = np.zeros((n,), dtype=np.float32)
    if n == 0:
        return probs

    t0 = time.time()
    for i0 in range(0, n, int(batch_size)):
        i1 = min(n, i0 + int(batch_size))
        batch = texts[i0:i1]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            out = model(**enc)
            logits = out.logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        probs[i0:i1] = p

        if sw is not None:
            frac = float(i1) / float(max(1, n))
            prog = base_progress + int(frac * float(span_progress))
            elapsed = time.time() - t0
            sw.update(
                stage="classifying",
                progress=min(99, max(0, prog)),
                message=f"{label}: {i1}/{n} rows (elapsed {int(elapsed)}s)",
                row_i=i1,
                row_n=n,
            )

    return probs


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True, help="Input transcript_sentences CSV")
    ap.add_argument("--out-csv", required=True, help="Output CSV with JAMA columns appended")
    ap.add_argument("--status-json", default=None, help="Path to write jama_classification_status.json")

    ap.add_argument(
        "--models-root",
        default="classifiers/jama_classifiers",
        help="Directory containing one subdir per label",
    )
    ap.add_argument(
        "--labels",
        default="f_anatomic,f_procedural,f_technical,f_visual_aid,f_praise,f_criticism",
        help="Comma-separated list of label directory names",
    )
    ap.add_argument("--text-col", default="sentence", help="Column to classify")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")

    args = ap.parse_args()

    app_root = Path(__file__).resolve().parent

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)

    status_path = Path(args.status_json) if args.status_json else out_csv.with_suffix(".jama_status.json")

    models_root = Path(args.models_root)
    if not models_root.is_absolute():
        models_root = app_root / models_root

    # Basic Torch import here (fine) but heavy Transformers imports happen later.
    import torch

    device, device_note = pick_device(args.device)
    dev = torch.device(device)

    labels = [s.strip() for s in (args.labels or "").split(",") if s.strip()]

    status = {
        "state": "running",
        "pid": os.getpid(),
        "started": _now_iso(),
        "stage": "init",
        "progress": 0,
        "message": "starting JAMA classification",
        "in_csv": str(in_csv),
        "out_csv": str(out_csv),
        "models_root": str(models_root),
        "labels": labels,
        "text_col": args.text_col,
        "device": device,
        "device_note": device_note,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "label": None,
        "row_i": 0,
        "row_n": None,
        "error": None,
        "finished": None,
        "updated": _now_iso(),
    }
    sw = StatusWriter(status_path, status, every_sec=0.5)
    sw.update(force=True)
    sw.update(message=f"starting JAMA classification on {device}" + (f" ({device_note})" if device_note else ""), force=True)

    try:
        sw.update(stage="load_csv", progress=1, message="loading input CSV", force=True)
        df = pd.read_csv(in_csv)
        if args.text_col not in df.columns:
            raise RuntimeError(f"Missing required text column {args.text_col!r} in {in_csv}")

        texts = df[args.text_col].fillna("").astype(str).tolist()
        n_rows = int(len(texts))
        sw.update(row_n=n_rows, force=True)

        # Heavy imports only when needed.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Spread progress 5..95 across labels.
        base = 5
        span = 90
        per_label = max(1, int(span / max(1, len(labels))))

        for li, label in enumerate(labels):
            label_dir = models_root / label
            thr_path = label_dir / "threshold.json"

            sw.update(
                stage="load_model",
                progress=min(99, base + li * per_label),
                message=f"loading model for {label}",
                label=label,
                model_dir=str(label_dir),
                threshold_path=str(thr_path),
                force=True,
            )

            if not label_dir.exists():
                raise RuntimeError(f"Model directory not found for {label!r}: {label_dir}")

            threshold = load_threshold(thr_path, label)

            tokenizer = AutoTokenizer.from_pretrained(str(label_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(label_dir))
            model.to(dev)
            model.eval()

            # Classify
            probs = predict_probs(
                texts,
                tokenizer,
                model,
                dev,
                batch_size=int(args.batch_size),
                max_length=int(args.max_length),
                sw=sw,
                label=label,
                base_progress=base + li * per_label,
                span_progress=per_label,
            )

            preds = (probs >= float(threshold)).astype(np.int32)
            conf = np.where(preds == 1, probs, 1.0 - probs)

            # Append columns
            df[label] = preds.astype(int)
            #df[f"prob_{label}"] = probs.astype(float)
            #df[f"conf_{label}"] = np.round(conf.astype(float), 2)

            sw.update(
                stage="classifying",
                progress=min(99, base + (li + 1) * per_label),
                message=f"{label}: done (threshold={threshold:.3f})",
                threshold=float(threshold),
                force=True,
            )

            # Free memory between labels (important on smaller GPUs)
            del model
            if device == "cuda":
                import torch

                torch.cuda.empty_cache()

        sw.update(stage="write", progress=99, message="writing output CSV", force=True)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        sw.update(
            state="done",
            stage="done",
            progress=100,
            message=f"done: wrote {len(df)} rows with {len(labels)} JAMA labels",
            finished=_now_iso(),
            force=True,
        )

    except Exception as e:
        sw.update(
            state="failed",
            stage="failed",
            progress=0,
            message=f"failed: {e}",
            error=str(e),
            finished=_now_iso(),
            force=True,
        )
        raise


if __name__ == "__main__":
    main()
