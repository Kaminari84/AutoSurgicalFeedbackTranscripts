import os
import socket
from datetime import datetime
from pathlib import Path
import json
import subprocess
import warnings
import sys
import uuid

# Clock OCR
import re
import numpy as np
import cv2
from PIL import Image
import pytesseract

import time
import shutil

import streamlit as st
from streamlit_autorefresh import st_autorefresh


APP_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = APP_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

AUDIO_DIR = APP_ROOT / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

JOBS_DIR = APP_ROOT / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

TMP_DIR = APP_ROOT / "_tmp"
TMP_DIR.mkdir(exist_ok=True)

_TIME_RE = re.compile(r'\b([01]?\d|2[0-3]):[0-5]\d:[0-5]\d\b')

# ----------------------------
# Download helpers
# ----------------------------
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # 200MB safety limit (adjust)

def _human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}TB"

def _resolve_job_path(path_str: str | None) -> Path | None:
    """
    Resolve paths coming from status.json:
    - absolute paths are used as-is
    - relative paths are assumed relative to APP_ROOT
    """
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = APP_ROOT / p
    return p

def download_row(label: str, path_str: str | None, *, key: str) -> None:
    p = _resolve_job_path(path_str)
    if p is None:
        return

    if not p.exists():
        st.caption(f"{label}: {path_str} (missing)")
        return

    size = p.stat().st_size
    try:
        shown = str(p.relative_to(APP_ROOT))
    except Exception:
        shown = str(p)

    c1, c2 = st.columns([8, 2])
    with c1:
        st.caption(f"{label}: `{shown}` ({_human_bytes(size)})")

    with c2:
        if size > MAX_DOWNLOAD_BYTES:
            st.caption("Too large (use rsync)")
            return

        # IMPORTANT: download_button will read from this file handle
        with open(p, "rb") as f:
            st.download_button(
                label="Download",
                data=f,
                file_name=p.name,
                key=key,
            )


def extract_first_frame_ffmpeg(video_path: Path, out_png: Path, t_sec: float = 0.0):
    # -ss before -i is fast; output a single PNG frame
    subprocess.check_call([
        "ffmpeg", "-y",
        "-ss", str(t_sec),
        "-i", str(video_path),
        "-frames:v", "1",
        "-vcodec", "png",
        str(out_png),
    ])

def preprocess_for_ocr(img_bgr: np.ndarray):
    scale = 3
    up = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    _, thr  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, ithr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return up, thr, ithr

def ocr_time_from_crop(crop_bgr: np.ndarray):
    up, thr, ithr = preprocess_for_ocr(crop_bgr)
    tess_cfg = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:'
    candidates = []
    for img in [up, thr, ithr]:
        txt = pytesseract.image_to_string(img, config=tess_cfg)
        candidates.append(txt)

    match = None
    for txt in candidates:
        m = _TIME_RE.search(txt)
        if m:
            match = m.group(0)
            break

    return match, candidates, up  # return upscaled image for debug display

def start_job(input_path: Path, clock_start: str | None = None, clock_roi: tuple[int,int,int,int] | None = None) -> Path:
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    log_path = job_dir / "worker.log"

    cmd = [
        sys.executable,
        str(APP_ROOT / "worker.py"),
        "--input", str(input_path),
        "--job-dir", str(job_dir),
    ]

    if clock_start is not None and clock_start.strip():
        cmd += ["--clock-start", clock_start.strip()]

    if clock_roi is not None:
        cmd += ["--clock-roi", ",".join(map(str, clock_roi))]


    with open(log_path, "w") as logf:
        subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return job_dir

def read_status(job_dir: Path):
    p = job_dir / "status.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

def job_started_ts(jd: Path) -> float:
    s = read_status_safe(jd)
    if s and s.get("started"):
        try:
            return datetime.fromisoformat(s["started"]).timestamp()
        except Exception:
            pass

    # fallback: parse "YYYYMMDD_HHMMSS_xxxxxxxx"
    try:
        parts = jd.name.split("_")
        if len(parts) >= 2:
            return datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y%m%d_%H%M%S").timestamp()
    except Exception:
        pass

    # last resort (may flip for running jobs)
    return jd.stat().st_mtime

def ffprobe_metadata(path: str) -> dict:
    """
    Returns basic video metadata using ffprobe (fast; no full decode).
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path
    ]
    out = subprocess.check_output(cmd, text=True)
    j = json.loads(out)

    # find first video stream
    vstreams = [s for s in j.get("streams", []) if s.get("codec_type") == "video"]
    v = vstreams[0] if vstreams else {}

    # FPS can be avg_frame_rate "30000/1001"
    fps = None
    afr = v.get("avg_frame_rate") or v.get("r_frame_rate")
    if afr and afr != "0/0" and "/" in afr:
        num, den = afr.split("/", 1)
        try:
            fps = float(num) / float(den)
        except Exception:
            fps = None

    # duration is usually in format.duration (string seconds)
    dur = None
    try:
        dur = float(j.get("format", {}).get("duration"))
    except Exception:
        dur = None

    return {
        "path": path,
        "duration_sec": dur,
        "fps": fps,
        "width": v.get("width"),
        "height": v.get("height"),
        "codec": v.get("codec_name"),
        "pix_fmt": v.get("pix_fmt"),
        "bit_rate": j.get("format", {}).get("bit_rate"),
        "size_bytes": j.get("format", {}).get("size"),
    }


st.set_page_config(page_title="TestWebApp", layout="wide")

st.title("Auto Transcription of Surgical Feedback")
st.caption(f"Time: {datetime.now().isoformat(timespec='seconds')} | Host: {socket.gethostname()}")

with st.sidebar:
    st.header("Controls")
    st.write("User:", os.getenv("USER", "unknown"))
    st.write("Upload dir:", str(UPLOAD_DIR))

    st.divider()
    st.subheader("Jobs")
    show_completed_jobs = st.checkbox("Show completed jobs", value=False)
    max_jobs_to_show = st.number_input("Max jobs to show", min_value=1, max_value=200, value=20)

    cleanup_hours = st.number_input("Delete completed jobs older than (hours)", min_value=0, max_value=24*30, value=24)
    do_cleanup = st.button("Delete old completed jobs")

st.subheader("1) Add video")

selected = None
tab1, tab2 = st.tabs(["Pick server-side file (recommended)", "Upload via browser (small tests)"])

with tab1:
    files = sorted(
        [p for p in UPLOAD_DIR.glob("*") if p.is_file()], 
        key=lambda p: p.stat().st_mtime, 
        reverse=True
    )
    if not files:
        st.warning(f"No files found in {UPLOAD_DIR}. Upload with rsync/scp, then refresh.")
        st.code(f"rsync -ah --info=progress2 <local_video.mp4> rafal@<DGX_IP>:{UPLOAD_DIR}/")
    else:
        selected = st.selectbox("Select a file already on the DGX", files, format_func=lambda p: p.name)
        if selected:
            st.write("Selected:", selected.name)
            st.write("Size (MB):", round(selected.stat().st_size / (1024**2), 2))

with tab2:
    up = st.file_uploader("Upload a small video", type=None)
    if up is not None:
        out_path = UPLOAD_DIR / up.name
        # Streamlit keeps uploaded file in memory/temp; write to disk
        with open(out_path, "wb") as f:
            f.write(up.getbuffer())
        st.success(f"Saved to {out_path}")
        st.write("Size (MB):", round(out_path.stat().st_size / (1024**2), 2))
        selected = out_path

# st.subheader("Selected file metadata")

# if selected is None:
#     st.info("Choose or upload a video first to see metadata.")
# else:
#     try:
#         meta = ffprobe_metadata(str(selected))
#         st.json(meta)
#     except FileNotFoundError:
#         st.error("ffprobe not found. Install with: conda install -c conda-forge ffmpeg -y")
#     except subprocess.CalledProcessError as e:
#         st.error(f"ffprobe failed: {e}")

# st.subheader("Torch / CUDA diagnostics")

# with st.expander("Open diagnostics", expanded=False):
#     try:
#         with warnings.catch_warnings(record=True) as w:
#             warnings.simplefilter("always")
#             import torch

#             st.write("torch:", torch.__version__)
#             st.write("torch.version.cuda:", torch.version.cuda)
#             st.write("cuda available:", torch.cuda.is_available())

#             if torch.cuda.is_available():
#                 st.write("device:", torch.cuda.get_device_name(0))
#                 st.write("capability:", torch.cuda.get_device_capability(0))

#                 x = torch.randn(512, 512, device="cuda")
#                 y = x @ x
#                 st.write("GPU matmul mean:", float(y.mean().item()))

#             # show warnings (including the GB10 capability message) in the UI
#             if w:
#                 st.warning("Warnings were raised during import/init:")
#                 for ww in w:
#                     st.code(str(ww.message))
#     except Exception as e:
#         st.error(f"PyTorch not ready: {e}")

## OCR PROCESS ##

st.subheader("2) Clock OCR (Real-time from Video)")

if selected is None:
    st.info("Choose a video first.")
else:
    # Extract first frame (cached to disk)
    frame_png = TMP_DIR / f"{selected.name}.firstframe.png"
    if not frame_png.exists():
        with st.spinner("Extracting first frame..."):
            extract_first_frame_ffmpeg(selected, frame_png, t_sec=0.0)

    img_pil = Image.open(frame_png).convert("RGB")
    img_np = np.array(img_pil)  # RGB, shape (H,W,3)
    H, W = img_np.shape[:2]

    # Default bottom-right crop (your Colab-ish numbers)
    default_h = min(150, H)
    default_w = min(530, W)
    default_top = max(0, H - default_h)
    default_left = max(0, W - default_w)

    st.caption("Use default bottom-right crop, or adjust the crop region, then OCR will run on the crop.")

    mode = st.radio(
        "Crop mode",
        ["Default bottom-right", "Manual sliders"],
        horizontal=True,
        index=0
    )

    if mode == "Default bottom-right":
        left, top = default_left, default_top
        width, height = default_w, default_h
    else:
        left = st.slider("Left (x)", 0, W - 1, default_left)
        top = st.slider("Top (y)", 0, H - 1, default_top)
        width = st.slider("Width", 1, W - left, default_w)
        height = st.slider("Height", 1, H - top, default_h)

    right = min(W, left + width)
    bottom = min(H, top + height)
    roi = (left, top, right, bottom)

    # Crop + OCR
    crop_rgb = img_np[top:bottom, left:right]
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    detected, candidates, up_dbg = ocr_time_from_crop(crop_bgr)

    col1, col2 = st.columns(2)
    with col1:
        st.write("ROI:", roi)
        st.image(crop_rgb, caption="Crop used for OCR", width='content')

    if detected:
        st.success(f"Detected time: {detected}")
    else:
        st.warning("No valid HH:MM:SS detected from OCR. Enter it manually below.")

    # Manual override (always available)
    manual = st.text_input(
        "Clock time on first frame (HH:MM:SS)",
        value=detected or "",
        placeholder="16:42:37"
    ).strip()

    # Persist for later steps/jobs
    if manual and _TIME_RE.fullmatch(manual):
        st.session_state["first_clock_str"] = manual
        st.session_state["clock_roi"] = roi
        st.info(f"Using clock start: {manual}")
    else:
        st.session_state.pop("first_clock_str", None)
        st.session_state["clock_roi"] = roi


## Video PROCESS ##
st.subheader("2) Process video (background job)")

if selected is None:
    st.info("Choose or upload a video first.")
else:
    clock_start = st.session_state.get("first_clock_str")
    clock_roi = st.session_state.get("clock_roi")

    if st.button("Start background job", type="primary"):
        jd = start_job(
            selected,
            clock_start=clock_start,
            clock_roi=clock_roi,
        )
        st.success(f"Started job: {jd.name}")

## List the jobs ##
st.subheader("Jobs (latest first)")

def read_status_safe(job_dir: Path):
    p = job_dir / "status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def is_completed(status: dict | None) -> bool:
    if not status:
        return False
    return status.get("state") in {"done", "failed", "canceled"}

# Optional: cleanup completed jobs older than cutoff
if do_cleanup:
    cutoff = time.time() - float(cleanup_hours) * 3600.0
    deleted = 0
    for jd in JOBS_DIR.glob("*"):
        if not jd.is_dir():
            continue
        s = read_status_safe(jd)
        if not s:
            continue
        if s.get("state") == "done":
            continue
        if is_completed(s) and jd.stat().st_mtime < cutoff:
            shutil.rmtree(jd, ignore_errors=True)
            deleted += 1
    st.success(f"Deleted {deleted} completed job(s).")

# Sort by job start time from internal status json
job_dirs = sorted(
    [p for p in JOBS_DIR.glob("*") if p.is_dir()],
    key=job_started_ts,
    reverse=True
)

auto_refresh = st.checkbox("Auto-refresh while running", value=True)

# Decide if we should refresh: only if any visible job is running
running_any = False
for jd in job_dirs[:10]:
    s = read_status(jd)
    if s and s.get("state") == "running":
        running_any = True
        break

if auto_refresh and running_any:
    # rerun page every 1s while something is running
    st_autorefresh(interval=1000, key="jobs_autorefresh")

# Filter out completed jobs by default (this is the “close once done” behavior)
if not show_completed_jobs:
    job_dirs = [jd for jd in job_dirs if not is_completed(read_status_safe(jd))]

job_dirs = job_dirs[: int(max_jobs_to_show)]

# ... keep everything above the loop the same ...

if not job_dirs:
    st.info("No jobs to show (toggle 'Show completed jobs' in the sidebar to see finished ones).")
else:
    for j_num, jd in enumerate(job_dirs, start=1):
        s = read_status_safe(jd)
        if not s:
            st.caption("No status yet…")
            continue

        # Display name: input filename if available, otherwise job id
        display_name = jd.name
        if s.get("input"):
            try:
                display_name = Path(s["input"]).name
            except Exception:
                display_name = str(s["input"])

        done = is_completed(s)
        state = s.get("state", "unknown")
        msg = s.get("message", "")

        # ---- Visual grouping (card if supported, else divider) ----
        try:
            box = st.container(border=True)
        except TypeError:
            box = st.container()

        with box:
            # Header
            header_cols = st.columns([0.8, 0.2])
            with header_cols[0]:
                st.write(f"{j_num}) Video: **{display_name}**")
                st.caption(f"Job: {jd.name}")
            with header_cols[1]:
                # compact state badge-like text
                st.caption(f"**{state}**")

            # Always show files (download area)
            with st.expander("Files", expanded=False):
                download_row("Raw audio", s.get("raw_audio_path"), key=f"dl-{jd.name}-rawwav")
                download_row("Denoised audio", s.get("denoised_audio_path"), key=f"dl-{jd.name}-denwav")
                download_row("Transcript (sentences)", s.get("transcript_sentences_csv"), key=f"dl-{jd.name}-trsent")

            # ---- Only show progress/status UI if NOT completed ----
            if not done:
                prog = int(s.get("progress", 0) or 0)
                prog = max(0, min(100, prog))
                st.progress(prog)
                st.caption(f"{state} — {msg}")

                # Stage sub-progress (0..100)
                stage = s.get("stage")
                stage_p = s.get("stage_progress")
                if stage and stage_p is not None:
                    stage_p = int(stage_p or 0)
                    stage_p = max(0, min(100, stage_p))
                    st.caption(f"Stage: {stage} — {stage_p}%")
                    st.progress(stage_p)

                # Segment counter (for transcription)
                seg_i = s.get("segment_i")
                seg_n = s.get("segment_n")
                if seg_n is not None:
                    try:
                        seg_n_i = int(seg_n)
                        seg_i_i = int(seg_i or 0)
                        if seg_n_i > 0:
                            st.caption(f"Segments: {seg_i_i}/{seg_n_i}")
                    except Exception:
                        pass
            else:
                # For completed jobs: keep it minimal
                # (optional) show the finished timestamp in a subtle way
                if s.get("finished"):
                    st.caption(f"Finished: {s['finished']}")

        # Separator between jobs (helps even with borders)
        #st.divider()


if selected is None:
    st.info("Choose or upload a video first.")
else:
    if st.button("Run processing"):
        # placeholder for your future pipeline
        st.write("Processing:", selected.name)
        st.success("Done (stub). Next: load model + run inference.")
