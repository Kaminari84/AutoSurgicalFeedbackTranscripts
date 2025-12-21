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
# Torchaudio patch for pyannote import on torchaudio 2.9+
# -------------------------
def _patch_torchaudio_for_pyannote() -> None:
    """
    pyannote.audio (3.x) imports torchaudio and calls APIs that were removed/deprecated
    around torchaudio 2.9 (e.g., list_audio_backends, AudioMetaData in some builds).

    This patch defines minimal shims so pyannote can import and use soundfile-based info.
    """
    import torchaudio  # must exist

    # list_audio_backends / get_audio_backend / set_audio_backend
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend=None: None  # type: ignore[attr-defined]

    # AudioMetaData
    if not hasattr(torchaudio, "AudioMetaData"):
        @dataclass
        class AudioMetaData:  # minimal fields commonly used
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int = 16
            encoding: str = "PCM_S"

        torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]

    # torchaudio.info (soundfile-backed)
    if not hasattr(torchaudio, "info"):
        import soundfile as sf

        def _info(path: str):
            f = sf.SoundFile(path)
            return torchaudio.AudioMetaData(  # type: ignore[attr-defined]
                sample_rate=int(f.samplerate),
                num_frames=int(len(f)),
                num_channels=int(f.channels),
                bits_per_sample=16,
                encoding="PCM_S",
            )

        torchaudio.info = _info  # type: ignore[attr-defined]


def diarize_pyannote(audio_path: Path, *, hf_token: str, min_speakers: int = 2) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _patch_torchaudio_for_pyannote()

    import torch
    from pyannote.audio import Pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    # If moving to CUDA fails in your env, fall back to CPU.
    try:
        pipe.to(device)
    except Exception:
        device = torch.device("cpu")

    ann = pipe(str(audio_path), min_speakers=min_speakers)

    rows: list[dict[str, Any]] = []
    for segment, _, speaker in ann.itertracks(yield_label=True):
        s = float(segment.start)
        e = float(segment.end)
        if e > s:
            rows.append({"start": s, "end": e, "speaker": str(speaker)})

    rows.sort(key=lambda r: (r["start"], r["end"]))

    meta = {
        "pipeline": "pyannote/speaker-diarization-3.1",
        "device": str(device),
        "min_speakers": int(min_speakers),
        "num_segments": len(rows),
    }
    return rows, meta


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
        "diarization_meta": None,
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
        # Step 3: Speaker diarization
        # -------------------------
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN is not set in the worker environment.")

        sw.update(stage="diarization", stage_progress=0, message="speaker diarization (pyannote)", progress=30, force=True)

        segments, diar_meta = diarize_pyannote(den_wav, hf_token=hf_token, min_speakers=2)

        seg_csv = job_dir / "segments.csv"
        pd.DataFrame(segments).to_csv(seg_csv, index=False)

        sw.update(
            stage="diarization",
            stage_progress=100,
            message=f"diarization done ({len(segments)} segments)",
            progress=60,
            segments_csv=str(seg_csv),
            num_segments=len(segments),
            diarization_meta=diar_meta,
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