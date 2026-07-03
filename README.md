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
│   ├── download_data.py      # Arctic Shift API streaming pipeline
│   ├── clean_data.py         # Text normalization and tokenization
│   ├── train_embeddings.py   # Per-year Word2Vec training
│   ├── align_embeddings.py   # Procrustes alignment to 2015 anchor
│   └── compute_drift.py      # Drift scores and neighbor extraction
├── app/
│   ├── backend/              # FastAPI server
│   └── frontend/             # React + Vite app
├── notebooks/
│   └── explore_drift.ipynb   # EDA and finding discovery
└── data/
    └── drift_results.json    # Precomputed drift data (Git LFS)
```

---

## Reproducing the results

```bash
git clone https://github.com/YOUR_USERNAME/semantic-drift-tracker
cd semantic-drift-tracker
bash setup.sh
python scripts/download_data.py   # ~2 hours
python scripts/clean_data.py
python scripts/train_embeddings.py
python scripts/align_embeddings.py
python scripts/compute_drift.py
```

---

## Limitations and future work

- Corpus limited to Reddit — findings reflect Reddit's demographic
  and cultural biases, not general language change
- Word2Vec captures distributional similarity but not word sense
  disambiguation — polysemous words may show artificial drift
- Future: extend to contextual embeddings (BERT) for sense-level
  drift detection; expand corpus to cover more domains
