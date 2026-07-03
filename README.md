# Semantic Drift Tracker

> Tracking how word meanings shift over time using unsupervised
> diachronic word embeddings trained on Reddit data (2015–2023).

**Live demo:** https://semantic-drift-tracker.vercel.app  
**API:** https://semantic-drift-api.onrender.com/docs

---

## What it does

This tool automatically detects semantic drift — changes in word
meaning over time — by training and aligning Word2Vec embeddings
on 2.7M Reddit comments across 9 years. Search any word to see
how its meaning and nearest neighbors evolved from 2015 to 2023.

---

## Key findings

Three distinct drift patterns were detected with no supervision
or labeled data:

**COVID-driven drift**
- `cough` — shifted from scattered, incoherent usage in 2015 to
  a tight symptom cluster (aches, vomiting, constipation) by 2023,
  reflecting sustained illness discourse post-2020
- `mask` — shifted from cosplay/gaming contexts (Majora's Mask,
  Halloween) in 2015 to protective equipment contexts
  (goggles, sprayed, brushed) by 2023

**Platform evolution drift**
- `pinned` — shifted from Pokémon battle terminology in 2015
  to Reddit moderation language (proposition, justifies, alignment)
  by 2023, tracking the rise of pinned posts as a platform feature
- `bold` — shifted from sports/strategy contexts in 2015
  to Reddit formatting vocabulary (stickied, megathread, weekly)
  by 2023

---

## Method

1. **Data collection** — 300,000 Reddit comments per year (2015–2023)
   streamed from the Arctic Shift API (~2.7M comments, ~84M tokens)
2. **Preprocessing** — lowercasing, contraction normalization,
   punctuation removal, single-character token filtering
3. **Embedding training** — Word2Vec (skip-gram, 200 dimensions,
   window=5, min_count=15) trained independently per year using gensim
4. **Procrustes alignment** — orthogonal rotation applied to each
   year's embedding space to align it to the 2015 reference space,
   making cross-year vector comparisons valid. Alignment quality:
   cosine similarity of top-100 words improved from 0.04–0.11
   (random) to 0.82–0.87 after alignment, with a gradual decay
   toward 2023 consistent with genuine semantic drift.
5. **Drift computation** — cosine distance between aligned vectors
   across years, computed for 13,099 shared vocabulary words
6. **Visualization** — FastAPI backend + React frontend with
   Recharts timeline charts and year-by-year neighbor tables

---

## Stack

| Layer | Technology |
|---|---|
| Data pipeline | Python, requests, zstandard |
| Embeddings | gensim Word2Vec |
| Alignment | scipy orthogonal Procrustes |
| Dimensionality reduction | UMAP |
| Backend | FastAPI, uvicorn |
| Frontend | React, Vite, Recharts |
| Deployment | Render (API), Vercel (frontend) |

---

## Repository structure

```
semantic-drift-tracker/
├── scripts/
│   ├── download_data.py       # Fetch 300k comments/year from Arctic Shift API
│   ├── clean_data.py          # Tokenize and normalize raw comments
│   ├── train_embeddings.py    # Train per-year Word2Vec models
│   ├── align_embeddings.py    # Procrustes alignment to 2015 reference space
│   └── compute_drift.py       # Cosine distance + neighbor computation
├── app/
│   ├── backend/
│   │   ├── main.py            # FastAPI: /word, /top-drift, /search, /health
│   │   └── run.py             # uvicorn entrypoint
│   └── frontend/
│       └── src/
│           ├── App.jsx        # Main UI: search, drift chart, neighbor table
│           └── api.js         # Fetch helpers (reads VITE_API_URL env var)
├── notebooks/
│   └── explore_drift.ipynb    # Interactive drift analysis
├── data/
│   ├── drift_results.json     # Per-word drift scores + neighbors (Git LFS)
│   └── top_drifted_words.json # Filtered top-200 most drifted words (Git LFS)
├── requirements.txt
├── runtime.txt                # Python 3.11.9 (Render)
└── .python-version            # Python 3.11.9 (pyenv)
```

---

## Reproducing the pipeline

```bash
# 1. Create virtualenv (venv placed outside repo to avoid MAX_PATH issues on Windows)
VENV_DIR=/c/venvs/sdt bash setup.sh

# 2. Activate
source /c/venvs/sdt/Scripts/activate   # Windows
# source /c/venvs/sdt/bin/activate     # macOS/Linux

# 3. Run pipeline in order
python scripts/download_data.py    # ~4–6 hours, skips completed years
python scripts/clean_data.py
python scripts/train_embeddings.py # ~20–40 min depending on hardware
python scripts/align_embeddings.py
python scripts/compute_drift.py

# 4. Start API
python app/backend/run.py

# 5. Start frontend
cd app/frontend && npm install && npm run dev
```

> **Note:** `data/raw/`, `data/processed/`, and `models/` are gitignored —
> run the pipeline locally to regenerate them.
> `data/drift_results.json` is stored via Git LFS and pulled automatically on clone.
