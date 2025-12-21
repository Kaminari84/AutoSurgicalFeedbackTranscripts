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

st.title("TestWebApp on DGX Spark")
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

st.subheader("Selected file metadata")

if selected is None:
    st.info("Choose or upload a video first to see metadata.")
else:
    try:
        meta = ffprobe_metadata(str(selected))
        st.json(meta)
    except FileNotFoundError:
        st.error("ffprobe not found. Install with: conda install -c conda-forge ffmpeg -y")
    except subprocess.CalledProcessError as e:
        st.error(f"ffprobe failed: {e}")

st.subheader("Torch / CUDA diagnostics")

with st.expander("Open diagnostics", expanded=False):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import torch

            st.write("torch:", torch.__version__)
            st.write("torch.version.cuda:", torch.version.cuda)
            st.write("cuda available:", torch.cuda.is_available())

            if torch.cuda.is_available():
                st.write("device:", torch.cuda.get_device_name(0))
                st.write("capability:", torch.cuda.get_device_capability(0))

                x = torch.randn(512, 512, device="cuda")
                y = x @ x
                st.write("GPU matmul mean:", float(y.mean().item()))

            # show warnings (including the GB10 capability message) in the UI
            if w:
                st.warning("Warnings were raised during import/init:")
                for ww in w:
                    st.code(str(ww.message))
    except Exception as e:
        st.error(f"PyTorch not ready: {e}")

## OCR PROCESS ##

st.subheader("Clock OCR (Step 1)")

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
        st.image(crop_rgb, caption="Crop used for OCR", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(up_dbg, cv2.COLOR_BGR2RGB), caption="Upscaled for OCR (debug)", use_container_width=True)
        st.write("OCR candidates:")
        for i, t in enumerate(candidates, 1):
            st.code(f"[{i}] {repr(t)}")

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
show_completed = st.checkbox("Show completed jobs", value=False)


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
        if (not show_completed) and s.get("state") == "done":
            continue
        if is_completed(s) and jd.stat().st_mtime < cutoff:
            shutil.rmtree(jd, ignore_errors=True)
            deleted += 1
    st.success(f"Deleted {deleted} completed job(s).")

# Sort by modification time (more accurate than reverse name)
job_dirs = sorted(
    [p for p in JOBS_DIR.glob("*") if p.is_dir()],
    key=lambda p: p.stat().st_mtime,
    reverse=True
)

# Filter out completed jobs by default (this is the “close once done” behavior)
if not show_completed_jobs:
    job_dirs = [jd for jd in job_dirs if not is_completed(read_status_safe(jd))]

job_dirs = job_dirs[: int(max_jobs_to_show)]

if not job_dirs:
    st.info("No jobs to show (toggle 'Show completed jobs' in the sidebar to see finished ones).")
else:
    for jd in job_dirs:
        st.write(f"**{jd.name}**")
        s = read_status_safe(jd)

        if not s:
            st.caption("No status yet…")
            continue

        # ----------------------------
        # Overall progress (0..100)
        # ----------------------------
        prog = int(s.get("progress", 0) or 0)
        prog = max(0, min(100, prog))
        st.progress(prog)

        state = s.get("state", "unknown")
        msg = s.get("message", "")
        st.caption(f"{state} — {msg}")

        # ----------------------------
        # Stage sub-progress (0..100)
        # ----------------------------
        stage = s.get("stage")
        stage_p = s.get("stage_progress")

        if stage and stage_p is not None:
            stage_p = int(stage_p or 0)
            stage_p = max(0, min(100, stage_p))
            st.caption(f"Stage: {stage} — {stage_p}%")
            st.progress(stage_p)

        # ----------------------------
        # Segment counter (for transcription)
        # ----------------------------
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

        with st.expander("Details", expanded=False):
            st.json(s)
            lp = jd / "worker.log"
            if lp.exists():
                st.code(lp.read_text()[-4000:])


if selected is None:
    st.info("Choose or upload a video first.")
else:
    if st.button("Run processing"):
        # placeholder for your future pipeline
        st.write("Processing:", selected.name)
        st.success("Done (stub). Next: load model + run inference.")
