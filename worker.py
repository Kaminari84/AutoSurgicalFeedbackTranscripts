import argparse
import csv
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

# ASR env + script
ASR_PYTHON = APP_ROOT / ".conda" / "asr" / "bin" / "python"

# Transcription
TRANSCRIBE_CLI = APP_ROOT / "transcribe_cli.py"

# -------------------------
# Classifiers
# -------------------------
CLASSIFY_CLI = APP_ROOT / "classify_cli.py"
CLASSIFIERS_ROOT = APP_ROOT / "classifiers"
THRESHOLDS_JSON  = CLASSIFIERS_ROOT / "thresholds.json"

ASR_DEVICE = os.getenv("ASR_DEVICE", "auto")  # auto|cuda|cpu

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
def _read_json(path: Path) -> dict | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except Exception:
        return None

def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing {label}: {path}")

def diarize_via_cli_with_progress(audio_path: Path, out_csv: Path, job_dir: Path, sw: StatusWriter, *, min_speakers: int = 2) -> None:
    diar_status = job_dir / "diarization_status.json"
    diar_log = job_dir / "diarize_cli.log"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(DIAR_PYTHON),
        "-u",  # unbuffered so logs flush quickly
        str(DIARIZE_CLI),
        "--audio", str(audio_path),
        "--out-csv", str(out_csv),
        "--min-speakers", str(int(min_speakers)),
        "--status-json", str(diar_status),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    sw.update(stage="diarization", stage_progress=0, message="starting diarization subprocess", progress=30, force=True,
              diarization_status_path=str(diar_status), diarization_log=str(diar_log))

    with open(diar_log, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)

        last_seen = None
        while True:
            rc = proc.poll()
            js = _read_json(diar_status)

            # Forward progress to Streamlit status.json
            if js:
                last_seen = js
                p = int(js.get("progress", 0) or 0)
                p = max(0, min(100, p))
                msg = js.get("message") or "diarizing…"
                stage = js.get("stage") or "diarization"

                # Map overall worker progress 30..60 during diarization
                overall = 30 + int((p / 100.0) * 30)

                sw.update(
                    stage="diarization",
                    stage_progress=p,
                    message=f"{stage}: {msg}",
                    progress=overall,
                )
            else:
                # No json yet — still provide heartbeat in UI
                sw.update(stage="diarization", message="diarizing… (waiting for diarization_status.json)", progress=31)

            if rc is not None:
                break
            time.sleep(0.5)

        if rc != 0:
            # Pull the best available structured error
            err = None
            if last_seen and last_seen.get("error"):
                err = last_seen["error"]
            raise RuntimeError(f"diarize_cli failed (rc={rc}). {err or 'See diarize_cli.log for details.'}")

# -------------------------
# Transcription via subprocess (isolated env)
# -------------------------
def transcribe_via_cli_with_progress(
    audio_path: Path,
    segments_csv: Path,
    out_seg_csv: Path,
    out_sent_csv: Path,
    job_dir: Path,
    sw: StatusWriter,
    *,
    model: str = "openai/whisper-large-v3",
    language: str = "en",
    task: str = "transcribe",
    device: Optional[str] = None,
) -> None:
    tr_status = job_dir / "transcription_status.json"
    tr_log = job_dir / "transcribe_cli.log"

    _require_file(ASR_PYTHON, "ASR_PYTHON")
    _require_file(TRANSCRIBE_CLI, "TRANSCRIBE_CLI")
    _require_file(segments_csv, "segments.csv")

    out_seg_csv.parent.mkdir(parents=True, exist_ok=True)
    out_sent_csv.parent.mkdir(parents=True, exist_ok=True)

    # Default to CPU unless you explicitly override (safer on GB10 until CUDA arch support is nailed).
    # Set ASR_DEVICE=cuda (or cpu) in your environment if you want to force.
    if device is None:
        device = os.getenv("ASR_DEVICE", "cpu")

    cmd = [
        str(ASR_PYTHON),
        "-u",
        str(TRANSCRIBE_CLI),
        "--audio", str(audio_path),
        "--segments", str(segments_csv),
        "--out-csv", str(out_seg_csv),
        "--out-sentences-csv", str(out_sent_csv),
        "--status-json", str(tr_status),
        "--model", model,
        "--language", language,
        "--task", task,
        "--device", ASR_DEVICE,
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    sw.update(
        stage="transcription",
        stage_progress=0,
        message="starting transcription",
        progress=60,
        force=True,
        transcription_status_path=str(tr_status),
        transcription_log=str(tr_log),
        transcript_segments_csv=str(out_seg_csv),
        transcript_sentences_csv=str(out_sent_csv),
        segment_i=0,
        segment_n=None,
    )

    with open(tr_log, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)

        last_seen = None
        while True:
            rc = proc.poll()
            js = _read_json(tr_status)

            if js:
                last_seen = js
                p = int(js.get("progress", 0) or 0)
                p = max(0, min(100, p))
                msg = js.get("message") or "transcribing…"

                seg_i = js.get("segment_i")
                seg_n = js.get("segment_n")

                # overall worker progress: 60..95
                overall = 60 + int((p / 100.0) * 35)

                sw.update(
                    stage="transcription",
                    stage_progress=p,
                    message=f"transcription: {msg}",
                    progress=overall,
                    segment_i=seg_i,
                    segment_n=seg_n,
                )
            else:
                sw.update(
                    stage="transcription",
                    stage_progress=0,
                    message="transcribing… (waiting for transcription_status.json)",
                    progress=61,
                )

            if rc is not None:
                break
            time.sleep(0.5)

        if rc != 0:
            err = None
            if last_seen and last_seen.get("error"):
                err = last_seen["error"]
            raise RuntimeError(f"transcribe_cli failed (rc={rc}). {err or 'See transcribe_cli.log'}")

    if not out_seg_csv.exists():
        raise RuntimeError(f"transcribe_cli finished but did not create: {out_seg_csv}")
    if not out_sent_csv.exists():
        raise RuntimeError(f"transcribe_cli finished but did not create: {out_sent_csv}")


# -------------------------
# Wall-clock augmentation (adds clock_* columns to transcript_sentences.csv)
# -------------------------
def add_wall_clock_columns_inplace(sent_csv: Path, first_clock_str: str) -> None:
    """
    Adds:
      - clock_start, clock_end (HH:MM:SS, modulo 24h)
      - day_offset_start, day_offset_end (0,1,2,...)

    Requires sent_csv to have start/end as video-relative HH:MM:SS strings.
    Implemented with stdlib only (no pandas dependency).
    """
    def hms_to_seconds(hms: str) -> int:
        parts = (hms or "").strip().split(":")
        if len(parts) != 3:
            return 0
        try:
            h, m, s = (int(float(x)) for x in parts)  # tolerate "00", "00.0"
        except Exception:
            return 0
        return h * 3600 + m * 60 + s

    def seconds_to_hms(sec: int) -> str:
        sec = int(sec)
        h = sec // 3600
        sec -= h * 3600
        m = sec // 60
        s = sec - m * 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    base_sec = hms_to_seconds(first_clock_str)

    tmp_path = sent_csv.with_suffix(".tmp")
    with sent_csv.open("r", newline="") as f_in, tmp_path.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            return
        fieldnames = list(reader.fieldnames)

        # Avoid duplicating columns if rerun
        for col in ["clock_start", "clock_end", "day_offset_start", "day_offset_end"]:
            if col not in fieldnames:
                fieldnames.append(col)

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            start_rel = hms_to_seconds(row.get("start", ""))
            end_rel = hms_to_seconds(row.get("end", ""))

            start_abs = base_sec + start_rel
            end_abs = base_sec + end_rel

            day_off_s = start_abs // 86400
            day_off_e = end_abs // 86400

            row["day_offset_start"] = str(day_off_s)
            row["day_offset_end"] = str(day_off_e)
            row["clock_start"] = seconds_to_hms(start_abs % 86400)
            row["clock_end"] = seconds_to_hms(end_abs % 86400)

            writer.writerow(row)

    tmp_path.replace(sent_csv)


def classify_irrelevant_via_cli_with_progress(
    in_sent_csv: Path,
    out_sent_classified_csv: Path,
    job_dir: Path,
    sw: StatusWriter,
    *,
    device: str = "auto",
    batch_size: int = 64,
    max_length: int = 256,
    label: str = "Irrelevant",
) -> None:
    """
    Runs classify_cli.py (in ASR env by default) and maps its progress to:
      overall progress: 95..99
      stage_progress:   0..100
    Produces:
      - out_sent_classified_csv
      - irr_classification_status.json
      - irr_classification.log
    """
    irr_status = job_dir / "irr_classification_status.json"
    irr_log = job_dir / "irr_classification.log"

    _require_file(CLASSIFY_CLI, "CLASSIFY_CLI")
    _require_file(CLASSIFIERS_ROOT, "CLASSIFIERS_ROOT")
    _require_file(THRESHOLDS_JSON, "THRESHOLDS_JSON")

    cmd = [
        str(ASR_PYTHON),
        str(CLASSIFY_CLI),
        "--in-csv", str(in_sent_csv),
        "--out-csv", str(out_sent_classified_csv),
        "--text-col", "sentence",
        "--status-json", str(irr_status),
        "--label", label,
        "--thresholds-json", str(THRESHOLDS_JSON),
        "--models-root", str(CLASSIFIERS_ROOT),
        "--device", device,
        "--batch-size", str(batch_size),
        "--max-length", str(max_length),
    ]

    job_dir.mkdir(parents=True, exist_ok=True)
    with irr_log.open("w") as logf:
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)

    last_msg = "starting irrelevant classification..."
    last_overall = None

    while True:
        st = _read_json(irr_status) or {}
        stage_prog = int(st.get("progress", 0) or 0)
        msg = st.get("message") or last_msg
        row_i = st.get("row_i", None)
        row_n = st.get("row_n", None)

        # map 0..100 -> 95..99 (leave 100 for pipeline "done")
        overall = 95 + int(4 * max(0, min(100, stage_prog)) / 100)

        if overall != last_overall or msg != last_msg:
            sw.update(
                stage="irr_classification",
                stage_progress=stage_prog,
                progress=overall,
                message=msg,
                irr_classification_status_path=str(irr_status),
                irr_classification_log=str(irr_log),
                transcript_sentences_classified_csv=str(out_sent_classified_csv),
                sentence_i=row_i,
                sentence_n=row_n,
            )
            last_overall = overall
            last_msg = msg

        if p.poll() is not None:
            break
        time.sleep(0.5)

    if p.returncode != 0:
        _require_file(irr_log, "irr_classification.log")
        tail = "\n".join(irr_log.read_text(errors="ignore").splitlines()[-40:])
        raise RuntimeError(f"Irrelevant classification failed (exit={p.returncode}). Log tail:\n{tail}")

    _require_file(out_sent_classified_csv, "transcript_sentences_classified.csv")
    sw.update(
        stage="irr_classification",
        stage_progress=100,
        progress=99,
        message="irrelevant classification done",
        irr_classification_status_path=str(irr_status),
        irr_classification_log=str(irr_log),
        transcript_sentences_classified_csv=str(out_sent_classified_csv),
        force=True,
    )

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
        "transcript_segments_csv": None,
        "transcript_sentences_csv": None,
        "transcript_sentences_classified_csv": None,
        "irr_classification_status_path": None,
        "irr_classification_log": None,
        "sentence_i": 0,
        "sentence_n": None,

        # logs/statuses
        "diarization_status_path": None,
        "diarization_log": None,
        "transcription_status_path": None,
        "transcription_log": None,

        # counters (for GUI)
        "segment_i": None,
        "segment_n": None,

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

        diarize_via_cli_with_progress(den_wav, seg_csv, job_dir, sw, min_speakers=2)

        # verify + load
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
        # Step 4: Transcription (ASR)
        # -------------------------
        out_seg_csv = job_dir / "transcript_segments.csv"
        out_sent_csv = job_dir / "transcript_sentences.csv"

        sw.update(stage="transcription", stage_progress=0, message="starting ASR", progress=60, force=True)

        transcribe_via_cli_with_progress(
            den_wav,
            seg_csv,
            out_seg_csv,
            out_sent_csv,
            job_dir,
            sw,
            model=os.getenv("ASR_MODEL", "openai/whisper-large-v3"),
            language=os.getenv("ASR_LANG", "en"),
            task=os.getenv("ASR_TASK", "transcribe"),
            device=None,
        )

        # finalize transcription stage
        
        # Optional: add wall-clock columns to transcript_sentences.csv using OCR time from first frame
        if args.clock_start:
            sw.update(stage="transcription", stage_progress=99, progress=94, message="adding wall-clock columns", force=True)
            add_wall_clock_columns_inplace(out_sent_csv, args.clock_start)
        sw.update(
            stage="transcription",
            stage_progress=100,
            message="transcription done",
            progress=95,
            transcript_segments_csv=str(out_seg_csv),
            transcript_sentences_csv=str(out_sent_csv),
            force=True,
        )

        # -------------------------
        # Step 5: Irrelevant classification (sentence-level)
        # -------------------------
        out_sent_classified_csv = job_dir / "transcript_sentences_classified.csv"
        sw.update(
            stage="irr_classification",
            stage_progress=0,
            message="starting irrelevant classification",
            progress=95,
            transcript_sentences_classified_csv=str(out_sent_classified_csv),
            irr_classification_status_path=str(job_dir / "irr_classification_status.json"),
            irr_classification_log=str(job_dir / "irr_classification.log"),
            force=True,
        )
        classify_irrelevant_via_cli_with_progress(
            in_sent_csv=out_sent_csv,
            out_sent_classified_csv=out_sent_classified_csv,
            job_dir=job_dir,
            sw=sw,
            device="auto",
        )
        # -------------------------

        # Mark done
        sw.update(
            state="done",
            stage="done",
            stage_progress=100,
            progress=100,
            message="completed (audio + diarization + transcription + irr classification)",
            finished=_now_iso(),
            force=True,
        )

    except Exception as e:
        status["state"] = "failed"
        status["error"] = str(e)
        status["message"] = f"failed: {e}"
        status["finished"] = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()