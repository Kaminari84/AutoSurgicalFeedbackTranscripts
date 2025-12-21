# diarize_cli.py
import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


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


def diarize(audio_path: Path, out_csv: Path, *, min_speakers: int, status_path: Path | None) -> int:
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
            sw.update(force=True, state="failed", stage="auth", progress=0, message="HF_TOKEN missing", error="HF_TOKEN missing")
        raise RuntimeError("HF_TOKEN is not set.")

    # Import inside function so the env is the diar env
    if sw:
        sw.update(stage="load_pipeline", progress=5, message="loading pyannote pipeline", force=True)

    from pyannote.audio import Pipeline

    # IMPORTANT: use `token=` (newer huggingface_hub); `use_auth_token` can break depending on hub version
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    if sw:
        sw.update(stage="load_pipeline", progress=15, message="pipeline loaded", force=True)

    # Heartbeat thread while the heavy call runs
    stop = {"flag": False}

    def heartbeat():
        t0 = time.time()
        # Heuristic: advance progress slowly up to ~90% while running
        # based on elapsed vs expected time. (Not perfect but very useful UX.)
        expected = None
        if dur is not None:
            # diarization can be slower than real-time; keep it conservative
            expected = max(60.0, dur * 2.0)  # seconds

        while not stop["flag"]:
            elapsed = time.time() - t0
            prog = 20  # baseline once we start running
            if expected:
                frac = min(1.0, elapsed / expected)
                prog = 20 + int(frac * 70)  # 20..90
            else:
                # no duration — just crawl upward slowly
                prog = min(90, 20 + int(elapsed / 5.0))

            if sw:
                sw.update(
                    stage="diarizing",
                    progress=prog,
                    message=f"diarizing… (elapsed {int(elapsed)}s)",
                    elapsed_sec=float(elapsed),
                )
            time.sleep(0.5)

    if sw:
        sw.update(stage="diarizing", progress=20, message="running diarization", force=True)

    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    hb_thread.start()

    try:
        ann = pipe(str(audio_path), min_speakers=int(min_speakers))
    finally:
        stop["flag"] = True
        hb_thread.join(timeout=2.0)

    if sw:
        sw.update(stage="writing_csv", progress=95, message="writing segments.csv", force=True)

    rows = []
    for segment, _, speaker in ann.itertracks(yield_label=True):
        s = float(segment.start)
        e = float(segment.end)
        if e > s:
            rows.append({"start": s, "end": e, "speaker": str(speaker)})

    rows.sort(key=lambda r: (r["start"], r["end"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

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
    args = ap.parse_args()

    audio = Path(args.audio)
    out_csv = Path(args.out_csv)
    status_path = Path(args.status_json) if args.status_json else None

    try:
        diarize(audio, out_csv, min_speakers=args.min_speakers, status_path=status_path)
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