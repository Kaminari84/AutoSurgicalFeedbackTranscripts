#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
import re


APP_ROOT = Path(__file__).resolve().parent
AUDIO_ROOT = APP_ROOT / "audio"
AUDIO_ROOT.mkdir(exist_ok=True)

TIME_RE = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d:[0-5]\d$")


# ----------------------------
# Atomic status writing (throttled)
# ----------------------------
def atomic_write_json(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


class StatusWriter:
    def __init__(self, status_path: Path, status: dict, every_sec: float = 0.5):
        self.status_path = status_path
        self.status = status
        self.every_sec = every_sec
        self.last_write = 0.0

    def update(self, *, force: bool = False, **fields):
        now = time.time()
        for k, v in fields.items():
            self.status[k] = v
        if force or (now - self.last_write) >= self.every_sec:
            atomic_write_json(self.status_path, self.status)
            self.last_write = now


# ----------------------------
# FFmpeg audio helpers
# ----------------------------
def run(cmd: list[str]):
    subprocess.check_call(cmd)


def ffmpeg_extract_audio(video_path: Path, out_wav: Path, sr: int = 16000, channels: int = 2):
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(video_path),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    run(cmd)


def ffmpeg_denoise(in_wav: Path, out_wav: Path):
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(in_wav),
        "-af", "afftdn",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="video path")
    ap.add_argument("--job-dir", required=True, help="job directory")
    ap.add_argument("--clock-start", default=None, help="HH:MM:SS from app.py")
    ap.add_argument("--clock-roi", default=None, help="left,top,right,bottom from app.py (stored only)")
    args = ap.parse_args()

    video_path = Path(args.input)
    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    status_path = job_dir / "status.json"

    clock_start = (args.clock_start or "").strip() or None
    if clock_start is not None and not TIME_RE.match(clock_start):
        # Don’t fail the job — just record it as invalid and continue.
        clock_start_invalid = clock_start
        clock_start = None
    else:
        clock_start_invalid = None

    status = {
        "state": "running",
        "pid": os.getpid(),
        "started": datetime.now().isoformat(timespec="seconds"),
        "input": str(video_path),

        "progress": 0,
        "stage": None,
        "stage_progress": 0,
        "message": "starting",

        # Step 1 provenance (record-only)
        "clock_start": clock_start,
        "clock_start_invalid": clock_start_invalid,
        "clock_roi": args.clock_roi,  # stored as string, no parsing here

        # Step 2 outputs
        "raw_audio_path": None,
        "denoised_audio_path": None,
    }

    atomic_write_json(status_path, status)
    sw = StatusWriter(status_path, status, every_sec=0.5)

    try:
        # -------------------------
        # Step 1: record-only
        # -------------------------
        sw.update(stage="clock", stage_progress=100, progress=10, message="clock params recorded", force=True)

        # -------------------------
        # Step 2: Extract + denoise audio
        # -------------------------
        sw.update(stage="audio", stage_progress=0, progress=15, message="extracting audio (raw)", force=True)

        audio_dir = AUDIO_ROOT / job_dir.name
        raw_wav = audio_dir / "audio_raw_16k_stereo.wav"
        den_wav = audio_dir / "audio_denoised_16k_stereo.wav"

        ffmpeg_extract_audio(video_path, raw_wav, sr=16000, channels=2)
        sw.update(raw_audio_path=str(raw_wav), stage_progress=60, progress=30, message="denoising audio", force=True)

        ffmpeg_denoise(raw_wav, den_wav)
        sw.update(denoised_audio_path=str(den_wav), stage_progress=100, progress=45, message="audio ready", force=True)

        # -------------------------
        # Placeholders for Steps 3–5
        # -------------------------
        sw.update(
            stage="pending",
            stage_progress=0,
            progress=45,
            message="TODO: Step 3 diarization, Step 4 transcription, Step 5 scoring + transcripts CSV",
            force=True,
        )

        sw.update(
            state="done",
            finished=datetime.now().isoformat(timespec="seconds"),
            progress=100,
            message="completed steps 1(record) + 2(audio).",
            force=True,
        )

    except Exception as e:
        sw.update(
            state="failed",
            error=str(e),
            finished=datetime.now().isoformat(timespec="seconds"),
            message=f"failed: {e}",
            force=True,
        )
        raise


if __name__ == "__main__":
    main()