#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf


_HMS_ONLY_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")

def parse_clock_start_to_sec(clock_start: str | None) -> int | None:
    """Parse HH:MM:SS into seconds since midnight. Returns None if invalid."""
    if not clock_start:
        return None
    s = clock_start.strip()
    if not _HMS_ONLY_RE.match(s):
        return None
    try:
        h, m, ss = s.split(":")
        return int(h) * 3600 + int(m) * 60 + int(ss)
    except Exception:
        return None

def wall_clock_from_offset(base_clock_sec: int, offset_sec: float) -> tuple[str, int]:
    """
    base_clock_sec: seconds since midnight at t=0 (from OCR)
    offset_sec: seconds since video start (float)
    Returns (HH:MM:SS modulo 24h, day_offset int).
    """
    tot = float(base_clock_sec) + max(0.0, float(offset_sec))
    day_offset = int(tot // 86400.0)
    mod = int(tot % 86400.0)  # drop sub-second for display to match overlay
    h = mod // 3600
    m = (mod % 3600) // 60
    s = mod % 60
    return f"{h:02d}:{m:02d}:{s:02d}", day_offset

# -------------------------
# Pretty time formatting
# -------------------------
def sec_to_hms(sec: float | None) -> str:
    """Format seconds-from-start into HH:MM:SS.mmm"""
    if sec is None:
        return ""
    try:
        x = float(sec)
    except Exception:
        return ""
    if math.isnan(x):
        return ""
    sign = "-" if x < 0 else ""
    x = abs(x)

    h = int(x // 3600)
    x -= h * 3600
    m = int(x // 60)
    x -= m * 60
    s = int(x)
    ms = int(round((x - s) * 1000.0))

    # handle rounding overflow (e.g., 1.9996 -> 2.000)
    if ms == 1000:
        ms = 0
        s += 1
    if s == 60:
        s = 0
        m += 1
    if m == 60:
        m = 0
        h += 1

    return f"{sign}{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def round_or_none(x: Any, ndigits: int) -> Any:
    try:
        if x is None:
            return None
        xf = float(x)
        if math.isnan(xf):
            return None
        return round(xf, ndigits)
    except Exception:
        return None


def confidence_from_avg_logprob(avg_logprob: float | None) -> float | None:
    """
    Turn avg_logprob into a [0,1] confidence-like score.
    Using exp(avg_logprob) = geometric mean token prob (simple + stable).
    """
    if avg_logprob is None:
        return None
    try:
        v = float(avg_logprob)
        if math.isnan(v):
            return None
        return float(math.exp(v))
    except Exception:
        return None


def pick_device(requested: Optional[str] = None) -> tuple[str, Optional[str]]:
    """
    Prefer CUDA by default, but only if it is actually usable.
    Returns (device, note). note is a human-readable warning if we had to fall back.
    """
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
# Sentence splitting + spans
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
    mono = wav.mean(axis=1)
    return mono, sr, duration


def estimate_noise_floor_rms(mono: np.ndarray, sr: int) -> float:
    frame_len = max(1, int(0.02 * sr))
    if mono.shape[0] < frame_len:
        return 1e-3
    n_frames = mono.shape[0] // frame_len
    x = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(x * x, axis=1) + 1e-12)
    return float(np.quantile(rms, 0.10))


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
    import torch

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

    seq = gen.sequences
    text = processor.batch_decode(seq, skip_special_tokens=True)[0].strip()

    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        return {"text": text, "avg_logprob": None, "avg_entropy": None}

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
        logits = outs.logits[0]

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
    i0 = max(0, int(start_sec * sr))
    i1 = min(mono.shape[0], int(end_sec * sr))
    seg = mono[i0:i1]
    if seg.size == 0:
        return {"text": "", "avg_logprob": None, "avg_entropy": None}

    if sr != 16000:
        from scipy.signal import resample_poly
        g = math.gcd(sr, 16000)
        up = 16000 // g
        down = sr // g
        seg = resample_poly(seg, up, down).astype(np.float32)
        sr2 = 16000
    else:
        seg = seg.astype(np.float32, copy=False)
        sr2 = sr

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
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--clock-start", default=None, help="HH:MM:SS from first video frame; used to emit clock_start/clock_end columns")
    ap.add_argument("--min-seg-sec", type=float, default=0.15)
    args = ap.parse_args()

    audio_path = Path(args.audio)
    seg_path = Path(args.segments)
    out_csv = Path(args.out_csv)
    out_sent = Path(args.out_sentences_csv) if args.out_sentences_csv else out_csv.with_name(out_csv.stem + "_sentences.csv")
    status_path = Path(args.status_json) if args.status_json else out_csv.with_suffix(".status.json")

    import torch

    device, device_note = pick_device(args.device)
    base_clock_sec = parse_clock_start_to_sec(args.clock_start)

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
        "clock_start": args.clock_start,
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
            start_sec = float(diar.at[i, "start"])
            end_sec = float(diar.at[i, "end"])
            speaker = str(diar.at[i, "speaker"])

            # Derived wall-clock fields (optional)
            if base_clock_sec is not None:
                seg_clock_start, seg_day_offset_start = wall_clock_from_offset(base_clock_sec, start_sec)
                seg_clock_end, seg_day_offset_end = wall_clock_from_offset(base_clock_sec, end_sec)
            else:
                seg_clock_start = seg_clock_end = None
                seg_day_offset_start = seg_day_offset_end = None

            # skip tiny segments
            if end_sec - start_sec < args.min_seg_sec:
                rows.append({
                    "seg_idx": i,
                    "speaker": speaker,
                    "start": sec_to_hms(start_sec),
                    "end": sec_to_hms(end_sec),
                    "clock_start": seg_clock_start,
                    "clock_end": seg_clock_end,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "signal_to_noise_ratio": None,
                    "transcription_confidence_score": None,
                    "audio_mag": 0.0,
                    "transcription": "",
                    "asr_avg_logprob": None,
                    "asr_avg_entropy": None,
                })
                continue

            # audio metrics
            i0 = max(0, int(start_sec * sr))
            i1 = min(mono.shape[0], int(end_sec * sr))
            seg = mono[i0:i1]
            rms = float(np.sqrt(np.mean(seg * seg) + eps)) if seg.size else 0.0
            snr_db = float(20.0 * math.log10((rms + eps) / (noise_rms + eps))) if seg.size else float("nan")

            tr = transcribe_segment_with_chunking(
                mono, sr, start_sec, end_sec,
                processor, model, dev,
            )
            text = (tr.get("text") or "").strip()
            avg_lp = tr.get("avg_logprob", None)
            avg_ent = tr.get("avg_entropy", None)

            conf = confidence_from_avg_logprob(avg_lp)
            conf = round_or_none(conf, 2)
            snr_out = round_or_none(snr_db, 2)

            rows.append({
                "seg_idx": i,
                "speaker": speaker,
                "start": sec_to_hms(start_sec),
                "end": sec_to_hms(end_sec),
                "clock_start": seg_clock_start,
                "clock_end": seg_clock_end,
                "signal_to_noise_ratio": snr_out,
                "transcription_confidence_score": conf,
                "audio_mag": rms,
                "transcription": text,
                "asr_avg_logprob": avg_lp,
                "asr_avg_entropy": avg_ent,
            })

            # sentence splitting (proportional fallback)
            spans = proportional_sentence_spans(text, seg_start_abs=start_sec, seg_end_abs=end_sec)
            for j, sp in enumerate(spans):
                s0 = float(sp["start"])
                s1 = float(sp["end"])
                if base_clock_sec is not None:
                    sent_clock_start, sent_day_offset_start = wall_clock_from_offset(base_clock_sec, s0)
                    sent_clock_end, sent_day_offset_end = wall_clock_from_offset(base_clock_sec, s1)
                else:
                    sent_clock_start = sent_clock_end = None
                    sent_day_offset_start = sent_day_offset_end = None

                sent_rows.append({
                    "seg_idx": i,
                    "sentence_i": j,
                    "speaker": speaker,
                    "start": sec_to_hms(s0),
                    "end": sec_to_hms(s1),
                    "clock_start": sent_clock_start,
                    "clock_end": sent_clock_end,
                    "sentence": sp["sentence"],
                    "signal_to_noise_ratio": snr_out,
                    "transcription_confidence_score": conf,
                })

            # progress update
            elapsed = time.time() - t0
            p = 5 + int(95 * (i + 1) / max(1, n))
            sw.update(
                stage="transcribing",
                progress=min(99, max(0, p)),
                segment_i=i + 1,
                message=f"transcribing… ({i+1}/{n}, elapsed {int(elapsed)}s)",
            )

            # write partials occasionally
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
