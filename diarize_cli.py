# diarize_cli.py
import argparse
import os
import pandas as pd
from pathlib import Path

def diarize(audio_path: Path, out_csv: Path, min_speakers: int = 2):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set")

    import torch
    from pyannote.audio import Pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    try:
        pipe.to(device)
    except Exception:
        pass

    ann = pipe(str(audio_path), min_speakers=min_speakers)

    rows = []
    for segment, _, speaker in ann.itertracks(yield_label=True):
        s = float(segment.start)
        e = float(segment.end)
        if e > s:
            rows.append({"start": s, "end": e, "speaker": str(speaker)})

    rows.sort(key=lambda r: (r["start"], r["end"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--min-speakers", type=int, default=2)
    args = ap.parse_args()

    n = diarize(Path(args.audio), Path(args.out_csv), min_speakers=args.min_speakers)
    print(f"Wrote {n} segments to {args.out_csv}")

if __name__ == "__main__":
    main()
