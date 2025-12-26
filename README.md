# AutoSurgicalFeedbackTranscripts

A Streamlit app that turns surgical videos into speaker-attributed transcripts (diarization + ASR) and exports downloadable artifacts (audio + CSVs). Designed to be DGX Spark friendly: background jobs, status tracking via JSON, and safe CPU fallback when CUDA isn’t usable.

## What it does

Given a video file, the app:
- (Optional) OCRs the on-screen clock from the first frame (for alignment/metadata).
- Extracts audio with FFmpeg (16 kHz stereo WAV).
- Denoises audio with FFmpeg (afftdn).
- Runs speaker diarization with pyannote/speaker-diarization-3.1.
- Runs Whisper transcription (Transformers) per diarized segment, and additionally splits into sentence-level rows.