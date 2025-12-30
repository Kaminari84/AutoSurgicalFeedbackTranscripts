#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd


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
        self.status.update(fields)
        self.status["updated"] = _now_iso()
        if force or (now - self._last) >= self.every_sec:
            atomic_write_json(self.path, self.status)
            self._last = now


def pick_device(requested: Optional[str] = None) -> tuple[str, Optional[str]]:
    req = (requested or "auto").strip().lower()
    import torch

    def cuda_probe() -> None:
        x = torch.randn((128, 128), device="cuda")
        _ = float((x @ x).mean().item())

    if req not in {"auto", "cuda", "cpu"}:
        return "cpu", f"unknown --device={requested!r}; using cpu"

    if req == "cpu":
        return "cpu", None

    if torch.version.cuda is None:
        return "cpu", "torch is CPU-only (torch.version.cuda=None)"

    if not torch.cuda.is_available():
        return "cpu", "torch.cuda.is_available() is False"

    try:
        cuda_probe()
        return "cuda", None
    except Exception as e:
        return "cpu", f"CUDA probe failed; falling back to CPU: {e}"


def _resolve_model_dir(models_root: Path, label: str) -> Path:
    # handles Irrelevant vs Irrelevent typo
    candidates = [
        models_root / label,
        models_root / "Irrelevant",
        models_root / label.lower(),
    ]
    for p in candidates:
        if p.exists():
            return p
    return models_root / label  # let caller raise


def load_hf_sequence_classifier(model_dir: Path, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(torch.device(device))
    model.eval()
    return tok, model


def predict_probs(
    texts: List[str],
    tok,
    model,
    device: str,
    *,
    batch_size: int = 32,
    max_length: int = 256,
) -> np.ndarray:
    import torch

    probs_all: List[np.ndarray] = []
    dev = torch.device(device)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(dev) for k, v in enc.items()}

        with torch.inference_mode():
            logits = model(**enc).logits
            if logits.shape[-1] == 1:
                p = torch.sigmoid(logits.squeeze(-1))
            else:
                p = torch.softmax(logits, dim=-1)[:, 1]

        probs_all.append(p.detach().float().cpu().numpy())

    return np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0,), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--text-col", default="sentence")
    ap.add_argument("--status-json", default=None)

    ap.add_argument("--label", default="Irrelevant")
    ap.add_argument("--thresholds-json", required=True)
    ap.add_argument("--models-root", required=True)

    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    status_path = Path(args.status_json) if args.status_json else out_csv.with_suffix(".status.json")

    label = args.label.strip()
    conf_col = f"conf_{label}"

    device, device_note = pick_device(args.device)

    status = {
        "state": "running",
        "pid": os.getpid(),
        "started": _now_iso(),
        "stage": "init",
        "progress": 0,
        "message": "starting classification",
        "in_csv": str(in_csv),
        "out_csv": str(out_csv),
        "text_col": args.text_col,
        "label": label,
        "thresholds_json": str(args.thresholds_json),
        "models_root": str(args.models_root),
        "device": device,
        "device_note": device_note,
        "row_i": 0,
        "row_n": None,
        "error": None,
        "finished": None,
        "updated": _now_iso(),
    }
    sw = StatusWriter(status_path, status, every_sec=0.5)
    sw.update(force=True)

    try:
        sw.update(stage="load_inputs", progress=2, message="reading input CSV", force=True)
        df = pd.read_csv(in_csv)
        if args.text_col not in df.columns:
            raise RuntimeError(f"--text-col {args.text_col!r} not found. Columns: {list(df.columns)}")

        texts = df[args.text_col].fillna("").astype(str).tolist()
        sw.update(row_n=int(len(texts)))

        sw.update(stage="load_thresholds", progress=5, message="loading thresholds", force=True)
        thr_map = json.loads(Path(args.thresholds_json).read_text())
        thr = thr_map.get(label, thr_map.get("Irrelevant", None))
        if thr is None:
            raise RuntimeError(f"threshold for {label!r} not found in {args.thresholds_json}")
        thr = float(thr)

        sw.update(stage="load_model", progress=10, message="loading classifier model", force=True)
        model_dir = _resolve_model_dir(Path(args.models_root), label)
        if not model_dir.exists():
            raise RuntimeError(f"model directory not found: {model_dir}")
        tok, model = load_hf_sequence_classifier(model_dir, device)

        sw.update(stage="predict", progress=20, message=f"classifying {len(texts)} rows", force=True)

        probs_chunks: List[np.ndarray] = []
        n = len(texts)
        bs = max(1, int(args.batch_size))

        for i in range(0, n, bs):
            chunk = texts[i : i + bs]
            probs = predict_probs(chunk, tok, model, device, batch_size=bs, max_length=int(args.max_length))
            probs_chunks.append(probs)

            done = min(n, i + len(chunk))
            p = 20 + int(75 * done / max(1, n))  # 20..95
            sw.update(stage="predict", progress=min(95, max(20, p)), row_i=done, message=f"{done}/{n}")

        probs_all = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.zeros((0,), dtype=np.float32)

        sw.update(stage="write", progress=97, message="writing output CSV", force=True)
        df[label] = (probs_all >= thr).astype(int)
        df[conf_col] = np.round(probs_all.astype(np.float32), 2)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        sw.update(state="done", stage="done", progress=100, message="done", finished=_now_iso(), force=True)

    except Exception as e:
        sw.update(state="failed", stage="failed", progress=0, message=f"failed: {e}", error=str(e), finished=_now_iso(), force=True)
        raise


if __name__ == "__main__":
    main()
