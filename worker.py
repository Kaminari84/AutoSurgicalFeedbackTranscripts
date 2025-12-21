import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

APP_ROOT = Path(__file__).resolve().parent
AUDIO_ROOT = APP_ROOT / "audio"
AUDIO_ROOT.mkdir(exist_ok=True)

# diarization env + script
DIAR_PYTHON = APP_ROOT / ".conda" / "diar" / "bin" / "python"
DIARIZE_CLI = APP_ROOT / "diarize_cli.py"

# -------------------------
# Status utilities
# -------------------------
def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


class StatusWriter:
    def __init__(self, status_path: Path, status: dict, every_sec: float = 0.5):
        self.status_path = status_path
        self.status = status
        self.every_sec = float(every_sec)
        self._last_write = 0.0

    def update(
        self,
        *,
        stage: Optional[str] = None,
        stage_progress: Optional[int] = None,
        message: Optional[str] = None,
        progress: Optional[int] = None,
        force: bool = False,
        **extra: Any,
    ) -> None:
        now = time.time()

        if stage is not None:
            self.status["stage"] = stage
        if stage_progress is not None:
            self.status["stage_progress"] = int(stage_progress)
        if message is not None:
            self.status["message"] = message
        if progress is not None:
            self.status["progress"] = int(progress)

        for k, v in extra.items():
            self.status[k] = v

        if force or (now - self._last_write) >= self.every_sec:
            atomic_write_json(self.status_path, self.status)
            self._last_write = now


# -------------------------
# FFmpeg helpers
# -------------------------
def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def ffmpeg_extract_audio(video_path: Path, out_wav: Path, sr: int = 16000, channels: int = 2) -> None:
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


def ffmpeg_denoise(in_wav: Path, out_wav: Path) -> None:
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



# -------------------------
# Diarization via subprocess (isolated env)
# -------------------------
def diarize_via_cli(audio_path: Path, out_csv: Path, *, min_speakers: int = 2) -> None:
    if not DIAR_PYTHON.exists():
        raise RuntimeError(f"Missing diarization python: {DIAR_PYTHON}")
    if not DIARIZE_CLI.exists():
        raise RuntimeError(f"Missing diarization script: {DIARIZE_CLI}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(DIAR_PYTHON),
        str(DIARIZE_CLI),
        "--audio", str(audio_path),
        "--out-csv", str(out_csv),
        "--min-speakers", str(int(min_speakers)),
    ]

    # inherit env so HF_TOKEN is visible
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    subprocess.check_call(cmd, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--clock-start", default=None)
    ap.add_argument("--clock-roi", default=None)
    args = ap.parse_args()

    video_path = Path(args.input)
    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    status_path = job_dir / "status.json"
    status = {
        "state": "running",
        "pid": os.getpid(),
        "started": datetime.now().isoformat(timespec="seconds"),
        "input": str(video_path),
        "progress": 0,
        "stage": None,
        "stage_progress": 0,
        "message": "starting",
        # carry-through from app.py
        "clock_start": args.clock_start,
        "clock_roi": args.clock_roi,
        # outputs
        "raw_audio_path": None,
        "denoised_audio_path": None,
        "segments_csv": None,
        "num_segments": None,
        "error": None,
        "finished": None,
    }
    atomic_write_json(status_path, status)
    sw = StatusWriter(status_path, status, every_sec=0.5)

    try:
        # -------------------------
        # Step 2: Extract + denoise audio
        # -------------------------
        sw.update(stage="audio", stage_progress=0, message="extracting audio (raw)", progress=5, force=True)

        audio_dir = AUDIO_ROOT / job_dir.name
        raw_wav = audio_dir / "audio_raw_16k_stereo.wav"
        den_wav = audio_dir / "audio_denoised_16k_stereo.wav"

        ffmpeg_extract_audio(video_path, raw_wav, sr=16000, channels=2)
        sw.update(stage="audio", stage_progress=50, message="denoising audio", progress=15, raw_audio_path=str(raw_wav), force=True)

        ffmpeg_denoise(raw_wav, den_wav)
        sw.update(stage="audio", stage_progress=100, message="audio ready", progress=25, denoised_audio_path=str(den_wav), force=True)

        # -------------------------
        # Step 3: Speaker diarization (runs in .conda/diar)
        # -------------------------
        if not os.getenv("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN is not set in the Streamlit/worker environment.")

        seg_csv = job_dir / "segments.csv"
        sw.update(stage="diarization", stage_progress=0, message="speaker diarization (pyannote)", progress=30, force=True)

        diarize_via_cli(den_wav, seg_csv, min_speakers=2)

        if not seg_csv.exists():
            raise RuntimeError(f"Diarization finished but output CSV not found: {seg_csv}")

        df = pd.read_csv(seg_csv)
        n = int(len(df))

        sw.update(
            stage="diarization",
            stage_progress=100,
            message=f"diarization done ({n} segments)",
            progress=60,
            segments_csv=str(seg_csv),
            num_segments=n,
            force=True,
        )

        # -------------------------
        # Placeholders for next steps
        # -------------------------
        sw.update(stage="transcription", stage_progress=0, message="(placeholder) transcription not implemented yet", progress=60, force=True)

        status["state"] = "done"
        status["progress"] = 100
        status["finished"] = datetime.now().isoformat(timespec="seconds")
        status["message"] = "completed (audio + diarization)"
        atomic_write_json(status_path, status)

    except Exception as e:
        status["state"] = "failed"
        status["error"] = str(e)
        status["message"] = f"failed: {e}"
        status["finished"] = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()