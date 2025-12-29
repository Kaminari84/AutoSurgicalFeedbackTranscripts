#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

def pick_device(requested: Optional[str] = None) -> tuple[str, Optional[str]]:
    """
    Prefer CUDA by default, but only if it is actually usable.
    Returns (device, note). note is a human-readable warning if we had to fall back.
    """
    req = (requested or "auto").strip().lower()

    import torch

    def cuda_probe() -> None:
        # Minimal op to confirm kernels actually work on this GPU
        x = torch.randn((256, 256), device="cuda")
        y = x @ x
        _ = float(y.mean().item())

    if req not in {"auto", "cuda", "cpu"}:
        return "cpu", f"unknown --device={requested!r}; using cpu"

    if req == "cpu":
        return "cpu", None

    # req is "cuda" or "auto"
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
        # CUDA exists but isn't usable (common w/ capability mismatch, missing kernels, bad driver pairing, etc.)
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
# Sentence splitting + spans (matches your Colab logic)
# -------------------------
_SENT_SPLIT_RE = r'(\s*[.?!…]+["”\']*\s+)'


def split_sentences_regex(text: str) -> List[str]:
    """Split text into sentences, keeping end punctuation."""
    import re

    if not text:
        return []
    parts = re.split(_SENT_SPLIT_RE, text)
    sents, buf = [], ""
    for chunk in parts:
        buf += chunk
        if re.search(r'\s*[.?!…]+["”\']*\s+$', chunk or ""):
            s = buf.strip()
            if s:
                sents.append(re.sub(r"\s+", " ", s))
            buf = ""
    if buf.strip():
        sents.append(re.sub(r"\s+", " ", buf.strip()))
    return [s for s in sents if s]


def proportional_sentence_spans(
    text: str,
    seg_start_abs: float,
    seg_end_abs: float,
) -> List[Dict[str, Any]]:
    sents = split_sentences_regex(text)
    if not sents:
        return []

    dur = max(0.0, float(seg_end_abs) - float(seg_start_abs))
    lens = np.array([max(1, len(s)) for s in sents], dtype=np.float64)
    weights = lens / lens.sum()

    out: List[Dict[str, Any]] = []
    cur = float(seg_start_abs)
    for s, w in zip(sents, weights):
        d = float(dur * w)
        s_start = cur
        s_end = cur + d
        out.append({"start": s_start, "end": s_end, "sentence": s})
        cur = s_end

    # snap last end exactly to seg_end_abs
    out[-1]["end"] = float(seg_end_abs)
    return out


# -------------------------
# Audio helpers
# -------------------------
def load_audio_mono(path: Path, *, dtype="float32") -> Tuple[np.ndarray, int, float]:
    """
    Returns mono waveform float32 in [-1, 1], sample_rate, duration_sec.
    """
    info = sf.info(str(path))
    sr = int(info.samplerate)
    frames = int(info.frames)
    duration = frames / float(sr)

    wav, _sr = sf.read(str(path), dtype=dtype, always_2d=True)
    # wav: (T, C)
    mono = wav.mean(axis=1)
    return mono, sr, duration


def estimate_noise_floor_rms(mono: np.ndarray, sr: int) -> float:
    """
    Roughly mirrors your Colab: 20ms frames, 10th percentile RMS. :contentReference[oaicite:2]{index=2}
    """
    frame_len = max(1, int(0.02 * sr))
    if mono.shape[0] < frame_len:
        return 1e-3
    n_frames = mono.shape[0] // frame_len
    x = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(x * x, axis=1) + 1e-12)
    return float(np.quantile(rms, 0.10))

# -------------------------
# Colab-style derived metrics
# -------------------------
def compute_signal_to_noise_ratio(snr_db: float | None) -> float | None:
    """Colab used: df['signal_to_noise_ratio'] = df['snr_db'].round(2)."""
    if snr_db is None:
        return None
    try:
        return float(round(float(snr_db), 2))
    except Exception:
        return None

def compute_transcription_confidence_score(
    avg_logprob: float | None,
    avg_entropy: float | None,
    *,
    entropy_max: float = 8.0,
) -> float | None:
    """
    Colab-style:
      transcription_confidence_score = exp(asr_avg_logprob) * (1 - clip(asr_avg_entropy/8, 0, 1))
    """
    if avg_logprob is None or avg_entropy is None:
        return None
    try:
        lp = float(avg_logprob)
        ent = float(avg_entropy)
    except Exception:
        return None

    ent_norm = float(np.clip(ent / float(entropy_max), 0.0, 1.0))
    score = float(np.exp(lp) * (1.0 - ent_norm))
    return score


# -------------------------
# Whisper (Transformers) transcription
# -------------------------
def load_whisper(model_name: str, device: str, language: str, task: str):
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    dev = torch.device(device)
    model.to(dev)

    # Match your Colab intent: set language + task to avoid surprises :contentReference[oaicite:3]{index=3}
    try:
        model.generation_config.language = language
        model.generation_config.task = task
    except Exception:
        pass

    model.eval()
    return processor, model, dev


def transcribe_chunk(
    audio_16k: np.ndarray,
    processor,
    model,
    device,
) -> Dict[str, Any]:
    """
    Returns: text + avg_logprob + avg_entropy (teacher-forced on generated seq)
    Mirrors your Colab pattern (generate -> teacher-forced logits). :contentReference[oaicite:4]{index=4}
    """
    import torch

    # processor wants float array (T,)
    inputs = processor(
        audio_16k,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        gen = model.generate(
            input_features=inputs["input_features"],
            attention_mask=inputs.get("attention_mask", None),
            do_sample=False,
            return_dict_in_generate=True,
        )

    seq = gen.sequences  # (1, T)
    text = processor.batch_decode(seq, skip_special_tokens=True)[0].strip()

    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        # Whisper tokenizer should have pad; fallback
        return {"text": text, "avg_logprob": None, "avg_entropy": None}

    # teacher-forced forward pass
    decoder_input_ids = seq[:, :-1].contiguous()
    labels = seq[:, 1:].contiguous()
    mask = labels != pad_id
    labels = labels.clone()
    labels[~mask] = -100

    with torch.inference_mode():
        outs = model(
            input_features=inputs["input_features"],
            attention_mask=inputs.get("attention_mask", None),
            decoder_input_ids=decoder_input_ids,
        )
        logits = outs.logits[0]  # (T-1, vocab)

    log_probs = torch.log_softmax(logits, dim=-1)
    tgt = labels[0].clone()
    tgt[tgt == -100] = 0
    token_logprobs = log_probs[torch.arange(tgt.size(0), device=log_probs.device), tgt]

    valid = mask[0]
    avg_logprob = float(token_logprobs[valid].mean().item()) if valid.any() else None

    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)
    avg_entropy = float(ent[valid].mean().item()) if valid.any() else None

    return {"text": text, "avg_logprob": avg_logprob, "avg_entropy": avg_entropy}


def transcribe_segment_with_chunking(
    mono: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    processor,
    model,
    device,
    *,
    chunk_s: float = 28.0,
    overlap_s: float = 2.0,
) -> Dict[str, Any]:
    """
    Whisper is happiest around ~30s windows; chunk long diar segments.
    Produces concatenated text and weighted avg metrics.
    """
    # slice
    i0 = max(0, int(start_sec * sr))
    i1 = min(mono.shape[0], int(end_sec * sr))
    seg = mono[i0:i1]
    if seg.size == 0:
        return {"text": "", "avg_logprob": None, "avg_entropy": None}

    # resample if needed (should already be 16k from your ffmpeg step, but keep safe)
    if sr != 16000:
        # lightweight polyphase resample via scipy
        from scipy.signal import resample_poly

        g = math.gcd(sr, 16000)
        up = 16000 // g
        down = sr // g
        seg = resample_poly(seg, up, down).astype(np.float32)
        sr2 = 16000
    else:
        seg = seg.astype(np.float32, copy=False)
        sr2 = sr

    # chunk
    if (i1 - i0) / sr2 <= chunk_s:
        return transcribe_chunk(seg, processor, model, device)

    hop = max(1, int((chunk_s - overlap_s) * sr2))
    win = max(1, int(chunk_s * sr2))

    texts: List[str] = []
    lp_sum = 0.0
    ent_sum = 0.0
    m_count = 0

    for s0 in range(0, seg.shape[0], hop):
        s1 = min(seg.shape[0], s0 + win)
        chunk = seg[s0:s1]
        if chunk.shape[0] < int(0.25 * sr2):
            break
        r = transcribe_chunk(chunk, processor, model, device)
        t = (r.get("text") or "").strip()
        if t:
            texts.append(t)
        if r.get("avg_logprob") is not None:
            lp_sum += float(r["avg_logprob"])
            m_count += 1
        if r.get("avg_entropy") is not None:
            ent_sum += float(r["avg_entropy"])

        if s1 >= seg.shape[0]:
            break

    out_text = " ".join(texts).strip()
    avg_logprob = (lp_sum / m_count) if m_count > 0 else None
    avg_entropy = (ent_sum / m_count) if m_count > 0 else None
    return {"text": out_text, "avg_logprob": avg_logprob, "avg_entropy": avg_entropy}


# -------------------------
# CLI main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="denoised wav path (16k preferred)")
    ap.add_argument("--segments", required=True, help="segments.csv with start,end,speaker")
    ap.add_argument("--out-csv", required=True, help="output transcript_segments.csv")
    ap.add_argument("--out-sentences-csv", default=None, help="output transcript_sentences.csv")
    ap.add_argument("--status-json", default=None, help="transcription_status.json path")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu (default: auto, prefers cuda)")
    ap.add_argument("--min-seg-sec", type=float, default=0.15)
    ap.add_argument("--entropy-max", type=float, default=8.0, help="normalizer for confidence score (default: 8.0)")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    seg_path = Path(args.segments)
    out_csv = Path(args.out_csv)
    out_sent = Path(args.out_sentences_csv) if args.out_sentences_csv else out_csv.with_name(out_csv.stem + "_sentences.csv")
    status_path = Path(args.status_json) if args.status_json else out_csv.with_suffix(".status.json")

    import torch  # still fine to import here

    device, device_note = pick_device(args.device)

    status = {
        "state": "running",
        "pid": os.getpid(),
        "started": _now_iso(),
        "stage": "init",
        "progress": 0,
        "message": "starting transcription",
        "audio_path": str(audio_path),
        "segments_csv": str(seg_path),
        "out_csv": str(out_csv),
        "out_sentences_csv": str(out_sent),
        "model": args.model,
        "device": device,
        "device_note": device_note,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "segment_i": 0,
        "segment_n": None,
        "audio_duration_sec": None,
        "error": None,
        "finished": None,
        "updated": _now_iso(),
    }
    sw = StatusWriter(status_path, status, every_sec=0.5)
    sw.update(force=True)
    sw.update(message=f"starting transcription on {device}" + (f" ({device_note})" if device_note else ""), force=True)

    try:
        sw.update(stage="load_audio", progress=1, message="loading audio", force=True)
        mono, sr, dur = load_audio_mono(audio_path, dtype="float32")
        noise_rms = estimate_noise_floor_rms(mono, sr)
        sw.update(audio_duration_sec=float(dur))

        sw.update(stage="load_model", progress=3, message=f"loading whisper model ({args.model})", force=True)
        processor, model, dev = load_whisper(args.model, device, args.language, args.task)

        diar = pd.read_csv(seg_path)
        for col in ["start", "end", "speaker"]:
            if col not in diar.columns:
                raise RuntimeError(f"segments.csv missing required column: {col}")

        diar = diar.sort_values(["start", "end"]).reset_index(drop=True)
        n = int(len(diar))
        sw.update(segment_n=n, stage="transcribing", progress=5, message=f"transcribing {n} diarized segments", force=True)

        rows: List[Dict[str, Any]] = []
        sent_rows: List[Dict[str, Any]] = []

        eps = 1e-12
        t0 = time.time()

        for i in range(n):
            start = float(diar.at[i, "start"])
            end = float(diar.at[i, "end"])
            speaker = str(diar.at[i, "speaker"])

            # skip tiny segments
            if end - start < args.min_seg_sec:
                snr_db = float("-inf")
                sig_noise = compute_signal_to_noise_ratio(snr_db)
                conf = None
                r = {
                    "seg_idx": i,
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "audio_mag": 0.0,
                    "snr_db": snr_db,
                    "signal_to_noise_ratio": sig_noise,
                    "transcription": "",
                    "asr_avg_logprob": None,
                    "asr_avg_entropy": None,
                    "transcription_confidence_score": conf,
                }
                rows.append(r)
                continue

            # audio metrics (like your Colab loop) :contentReference[oaicite:5]{index=5}
            i0 = max(0, int(start * sr))
            i1 = min(mono.shape[0], int(end * sr))
            seg = mono[i0:i1]
            rms = float(np.sqrt(np.mean(seg * seg) + eps)) if seg.size else 0.0
            snr_db = float(20.0 * math.log10((rms + eps) / (noise_rms + eps))) if seg.size else float("-inf")

            tr = transcribe_segment_with_chunking(
                mono, sr, start, end,
                processor, model, dev,
            )
            text = (tr.get("text") or "").strip()
            avg_lp = tr.get("avg_logprob", None)
            avg_ent = tr.get("avg_entropy", None)

            sig_noise = compute_signal_to_noise_ratio(snr_db)
            conf = compute_transcription_confidence_score(avg_lp, avg_ent, entropy_max=float(args.entropy_max))

            r = {
                "seg_idx": i,
                "speaker": speaker,
                "start": start,
                "end": end,
                "audio_mag": rms,
                "snr_db": snr_db,
                "signal_to_noise_ratio": sig_noise,
                "transcription": text,
                "asr_avg_logprob": avg_lp,
                "asr_avg_entropy": avg_ent,
                "transcription_confidence_score": conf,
            }
            rows.append(r)

            # sentence splitting (proportional fallback) :contentReference[oaicite:6]{index=6}
            spans = proportional_sentence_spans(text, seg_start_abs=start, seg_end_abs=end)
            for j, sp in enumerate(spans):
                sent_rows.append({
                    "seg_idx": i,
                    "sentence_i": j,
                    "speaker": speaker,
                    "start": float(sp["start"]),
                    "end": float(sp["end"]),
                    "sentence": sp["sentence"],
                    # propagate per-segment metrics to each sentence row (Colab-style)
                    "signal_to_noise_ratio": sig_noise,
                    "transcription_confidence_score": conf,
                })

            # progress update
            elapsed = time.time() - t0
            p = 5 + int(95 * (i + 1) / max(1, n))  # keep 0..100 with headroom
            sw.update(
                stage="transcribing",
                progress=min(99, max(0, p)),
                segment_i=i + 1,
                message=f"transcribing… ({i+1}/{n}, elapsed {int(elapsed)}s)",
            )

            # write partials occasionally (nice for debugging)
            if (i + 1) % 10 == 0:
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(str(out_csv) + ".part.csv", index=False)
                pd.DataFrame(sent_rows).to_csv(str(out_sent) + ".part.csv", index=False)

        sw.update(stage="write", progress=99, message="writing CSV outputs", force=True)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        pd.DataFrame(sent_rows).to_csv(out_sent, index=False)

        sw.update(
            state="done",
            stage="done",
            progress=100,
            message=f"done: wrote {len(rows)} segments + {len(sent_rows)} sentences",
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
