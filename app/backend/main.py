"""FastAPI backend serving semantic drift data computed by scripts/compute_drift.py."""

import bisect
import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"

# Load once at import time — data is static, ~50 MB total, fits comfortably in RAM
with open(DATA_DIR / "drift_results.json", "r", encoding="utf-8") as _f:
    drift_results: dict = json.load(_f)

with open(DATA_DIR / "top_drifted_words.json", "r", encoding="utf-8") as _f:
    top_drifted: list = json.load(_f)

# Sorted word list for O(log n) prefix search
sorted_words: list[str] = sorted(drift_results.keys())

app = FastAPI(title="Semantic Drift Tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "words_loaded": len(drift_results)}


@app.get("/word/{word}")
def get_word(word: str):
    key = word.lower()
    if key not in drift_results:
        raise HTTPException(status_code=404, detail={"error": "word not found"})
    data = drift_results[key]
    return {
        "word": key,
        "yearly_drift": data["yearly_drift"],
        "cumulative_drift": data["cumulative_drift"],
        "neighbors": data["neighbors"],
    }


@app.get("/top-drift")
def top_drift(limit: Annotated[int, Query(ge=1, le=200)] = 50):
    return [
        {"word": word, "drift_score": score}
        for word, score in top_drifted[:limit]
    ]


@app.get("/search")
def search(q: str = ""):
    query = q.lower()
    if not query:
        return []
    start = bisect.bisect_left(sorted_words, query)
    results = []
    for word in sorted_words[start:]:
        if not word.startswith(query):
            break
        results.append({"word": word})
        if len(results) == 10:
            break
    return results
