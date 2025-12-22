#!/usr/bin/env python3
# diarize_cli.py
import argparse
import inspect
import json
import os
import threading
import time
import warnings
import math

from datetime import datetime
from pathlib import Path

import pandas as pd


def configure_warnings(suppress: bool = True) -> None:
    """Silence repeated TorchAudio/pyannote deprecation warnings."""
    if not suppress:
        return

    warnings.filterwarnings(
        "ignore",
        message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*torchaudio\._backend\.utils\.info has been deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*torchaudio\._backend\.common\.AudioMetaData has been deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*In 2\.9, this function's implementation will be changed to use torchaudio\.load_with_torchcodec.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*std\(\): degrees of freedom is <= 0.*",
        category=UserWarning,
    )


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


class StatusWriter:
    def __init__(self, path: Path, every_sec: float = 0.5):
        self.path = path
        self.every_sec = float(every_sec)
        self._last = 0.0
        self.state: dict = {}

    def update(self, *, force: bool = False, **fields):
        now = time.time()
        self.state.update(fields)
        self.state["updated"] = datetime.now().isoformat(timespec="seconds")
        if force or (now - self._last) >= self.every_sec:
            atomic_write_json(self.path, self.state)
            self._last = now


def get_audio_duration_sec(audio_path: Path) -> float | None:
    # Lightweight duration read; avoids torchaudio entirely
    try:
        import soundfile as sf
        with sf.SoundFile(str(audio_path)) as f:
            return float(len(f)) / float(f.samplerate)
    except Exception:
        return None


def diarize(
    audio_path: Path,
    out_csv: Path,
    *,
    min_speakers: int,
    status_path: Path | None,
    suppress_warnings: bool = True,
    prefer_gpu: bool = True,
    show_console_progress: bool = True,
) -> int:
    # IMPORTANT: configure warnings BEFORE importing pyannote/torchaudio stack
    configure_warnings(suppress_warnings)

    sw = StatusWriter(status_path, every_sec=0.5) if status_path else None

    pid = os.getpid()
    dur = get_audio_duration_sec(audio_path)

    if sw:
        sw.update(
            force=True,
            state="running",
            pid=pid,
            started=datetime.now().isoformat(timespec="seconds"),
            stage="init",
            progress=0,
            message="starting diarize_cli",
            audio_path=str(audio_path),
            out_csv=str(out_csv),
            min_speakers=int(min_speakers),
            audio_duration_sec=dur,
        )

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        if sw:
            sw.update(force=True, state="failed", stage="auth", progress=0,
                      message="HF_TOKEN missing", error="HF_TOKEN missing")
        raise RuntimeError("HF_TOKEN is not set.")

    if sw:
        sw.update(stage="load_pipeline", progress=5, message="loading pyannote pipeline", force=True)

    # Imports inside function so we use the currently-activated env
    import torch
    from torch.torch_version import TorchVersion
    import pyannote.audio.core.task as task_mod
    from pyannote.audio import Pipeline

    # Choose device
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Allowlist classes needed by pyannote checkpoints under PyTorch weights_only safety
    safe = [TorchVersion]
    safe += [
        obj for obj in vars(task_mod).values()
        if inspect.isclass(obj) and obj.__module__ == task_mod.__name__
    ]
    torch.serialization.add_safe_globals(safe)

    # HF hub pinned in your env, so use_auth_token is fine.
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipe.to(device)

    if sw:
        sw.update(
            stage="load_pipeline",
            progress=15,
            message=f"pipeline loaded (device={device.type})",
            device=device.type,
            torch_version=str(torch.__version__),
            cuda_available=bool(torch.cuda.is_available()),
            force=True,
        )

    def estimate_expected_sec(dur_sec: float | None) -> tuple[float | None, float | None]:
        """
        Estimate diarization wall time (sec) from audio duration (sec) using GPU-derived fit.

        Matches your observed points:
        dur=550.485  -> expected ~22.11
        dur=4936.021 -> expected ~300.20

        Returns (expected_sec, rtf) where rtf = expected_sec / dur_sec.
        """
        if dur_sec is None:
            return None, None

        d = max(1.0, float(dur_sec))
        # log-duration fit (derived from your two measurements)
        rtf = -0.0197 + 0.00947 * math.log(d)

        # keep it sane across weird edge cases
        rtf = max(0.02, min(0.08, rtf))

        expected = max(8.0, d * rtf)  # avoid tiny/zero expected
        return expected, rtf


    # Heartbeat thread while the heavy call runs (updates status JSON)
    stop = {"flag": False}

    def heartbeat():
        t0 = time.time()

        expected, rtf = estimate_expected_sec(dur)
        # Keep progress monotonic
        last_prog = 20

        while not stop["flag"]:
            elapsed = time.time() - t0

            if expected is not None:
                frac = min(1.0, elapsed / expected)
                prog = 20 + int(frac * 70)  # 20..90
                eta = max(0.0, expected - elapsed)
                msg = f"diarizing… (elapsed {int(elapsed)}s, eta {int(eta)}s, rtf≈{rtf:.3f})"
            else:
                # no duration — crawl upward slowly
                prog = min(90, 20 + int(elapsed / 5.0))
                msg = f"diarizing… (elapsed {int(elapsed)}s)"

            prog = max(last_prog, min(90, prog))
            last_prog = prog

            if sw:
                sw.update(
                    stage="diarizing",
                    progress=prog,
                    message=msg,
                    elapsed_sec=float(elapsed),
                    expected_sec=(float(expected) if expected is not None else None),
                    runtime_factor=(float(rtf) if rtf is not None else None),
                )
            time.sleep(0.5)

    if sw:
        sw.update(stage="diarizing", progress=20, message="running diarization", force=True)

    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    hb_thread.start()

    try:
        # Optional console progress (tqdm) using pyannote ProgressHook
        if show_console_progress:
            from pyannote.audio.pipelines.utils.hook import ProgressHook
            with ProgressHook() as hook:
                ann = pipe(str(audio_path), min_speakers=int(min_speakers), hook=hook)
        else:
            ann = pipe(str(audio_path), min_speakers=int(min_speakers))
    finally:
        stop["flag"] = True
        hb_thread.join(timeout=2.0)

    if sw:
        sw.update(stage="writing_csv", progress=95, message="writing diarization CSV", force=True)

    rows = []
    for segment, _, speaker in ann.itertracks(yield_label=True):
        s = float(segment.start)
        e = float(segment.end)
        if e > s:
            rows.append({"start": s, "end": e, "speaker": str(speaker)})

    rows.sort(key=lambda r: (r["start"], r["end"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["start", "end", "speaker"]).to_csv(out_csv, index=False)

    if sw:
        sw.update(
            force=True,
            state="done",
            stage="done",
            progress=100,
            message=f"done — wrote {len(rows)} segments",
            num_segments=len(rows),
            finished=datetime.now().isoformat(timespec="seconds"),
        )

    print(f"Wrote {len(rows)} segments to {out_csv}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--min-speakers", type=int, default=2)
    ap.add_argument("--status-json", default=None)

    # Keep your UX knobs (optional; defaults match your current behavior)
    ap.add_argument("--show-warnings", action="store_true", help="Show warnings (default: suppressed)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (default: use GPU if available)")
    ap.add_argument("--no-console-progress", action="store_true", help="Disable console progress bar")

    args = ap.parse_args()

    audio = Path(args.audio)
    out_csv = Path(args.out_csv)
    status_path = Path(args.status_json) if args.status_json else None

    try:
        diarize(
            audio,
            out_csv,
            min_speakers=args.min_speakers,
            status_path=status_path,
            suppress_warnings=(not args.show_warnings),
            prefer_gpu=(not args.cpu),
            show_console_progress=(not args.no_console_progress),
        )
    except Exception as e:
        if status_path:
            sw = StatusWriter(status_path, every_sec=0.0)
            sw.update(
                force=True,
                state="failed",
                stage="failed",
                progress=0,
                message=f"failed: {e}",
                error=str(e),
                finished=datetime.now().isoformat(timespec="seconds"),
            )
        raise


if __name__ == "__main__":
    main()