import argparse
import json
import os
import time
import subprocess
from datetime import datetime
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
AUDIO_ROOT = APP_ROOT / "audio"
AUDIO_ROOT.mkdir(exist_ok=True)

def write_status(p: Path, d: dict):
    p.write_text(json.dumps(d, indent=2))

def run(cmd: list[str]):
    # Keep logs clean; ffmpeg still prints errors if something breaks
    return subprocess.check_call(cmd)


def ffmpeg_extract_audio(video_path: Path, out_wav: Path, sr: int = 16000, channels: int = 2):
    """
    Extract audio track to PCM WAV at sr Hz. Keeps stereo by default.
    """
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
    """
    Simple denoise using ffmpeg's afftdn filter.
    You can tune later; this is a reasonable starter.
    """
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


def parse_roi(s: str):
    # "l,t,r,b"
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("clock-roi must be 'left,top,right,bottom'")
    return tuple(parts)


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
        "message": "starting",
        "clock_start": args.clock_start,
        "clock_roi": None,
    }

    if args.clock_roi:
        try:
            status["clock_roi"] = parse_roi(args.clock_roi)
        except Exception as e:
            status["clock_roi_error"] = str(e)

    write_status(status_path, status)

    try:
        # -------------------------
        # Step 2: Extract + denoise audio
        # -------------------------
        status["message"] = "extracting audio (raw)"
        status["progress"] = 5
        write_status(status_path, status)

        audio_dir = AUDIO_ROOT / job_dir.name
        raw_wav = audio_dir / "audio_raw_16k_stereo.wav"
        den_wav = audio_dir / "audio_denoised_16k_stereo.wav"

        ffmpeg_extract_audio(video_path, raw_wav, sr=16000, channels=2)

        status["message"] = "denoising audio"
        status["progress"] = 15
        status["raw_audio_path"] = str(raw_wav)
        write_status(status_path, status)

        ffmpeg_denoise(raw_wav, den_wav)

        status["denoised_audio_path"] = str(den_wav)
        status["message"] = "audio ready"
        status["progress"] = 20
        write_status(status_path, status)

        # -------------------------
        # Keep your placeholders for now
        # (we’ll replace these with diarization/transcription next)
        # -------------------------
        stages = [
            ("speaker diarization", 40),
            ("transcribe segments", 70),
            ("score + write CSV", 95),
        ]
        for msg, prog in stages:
            status["message"] = msg
            status["progress"] = prog
            write_status(status_path, status)
            time.sleep(1)

        # torch smoke test (optional)
        try:
            import torch
            status["torch"] = torch.__version__
            status["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                status["device"] = torch.cuda.get_device_name(0)
                x = torch.randn(512, 512, device="cuda")
                y = x @ x
                status["gpu_test_mean"] = float(y.mean().item())
        except Exception as e:
            status["torch_error"] = str(e)

        status["state"] = "done"
        status["progress"] = 100
        status["finished"] = datetime.now().isoformat(timespec="seconds")
        status["message"] = "completed"
        write_status(status_path, status)

    except Exception as e:
        status["state"] = "failed"
        status["error"] = str(e)
        status["finished"] = datetime.now().isoformat(timespec="seconds")
        write_status(status_path, status)
        raise


if __name__ == "__main__":
    main()