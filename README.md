# AutoSurgicalFeedbackTranscripts

Streamlit-based app for generating surgical feedback transcripts and quality metrics from uploaded videos (DGX Spark friendly).

## Run locally (on DGX)
```bash
conda activate ./.conda/env
streamlit run app.py --server.address 0.0.0.0 --server.port 8501