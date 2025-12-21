import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

def write_status(p: Path, d: dict):
    p.write_text(json.dumps(d, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--job-dir", required=True)
    args = ap.parse_args()

    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    status_path = job_dir / "status.json"

    status = {
        "state": "running",
        "pid": os.getpid(),
        "started": datetime.now().isoformat(timespec="seconds"),
        "input": args.input,
        "progress": 0,
        "message": "starting",
    }
    write_status(status_path, status)

    try:
        stages = ["ffprobe", "decode", "run model", "postprocess"]
        for i, msg in enumerate(stages, start=1):
            status["message"] = msg
            status["progress"] = int((i - 1) / len(stages) * 100)
            write_status(status_path, status)
            time.sleep(2)  # replace with real work

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
