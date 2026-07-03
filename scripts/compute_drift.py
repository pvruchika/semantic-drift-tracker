"""Compute per-word semantic drift across years from Procrustes-aligned embeddings.

Drift is measured as cosine distance (1 - cosine similarity) between a word's
vector in different years. This is only meaningful because align_embeddings.py
already rotated every year's space into the same 2015-anchored coordinate
system -- without that step the per-year vectors aren't comparable at all.
"""

import json
import os

import numpy as np

try:
    from nltk.corpus import words as nltk_words
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    from nltk.corpus import words as nltk_words

try:
    ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())
except LookupError:
    import nltk
    nltk.download("words")
    ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())

MODELS_DIR = "models"
DATA_DIR = "data"
YEARS = list(range(2015, 2024))
TOP_K_NEIGHBORS = 10
TOP_N_DRIFTED = 200
NEIGHBOR_BATCH = 500
MIN_WORD_LENGTH = 4
MIN_DRIFT_FOR_OOV = 0.5
MIN_LENGTH_FOR_OOV = 5

DRIFT_RESULTS_PATH = os.path.join(DATA_DIR, "drift_results.json")
TOP_DRIFTED_PATH = os.path.join(DATA_DIR, "top_drifted_words.json")
TOP_DRIFTED_UNFILTERED_PATH = os.path.join(DATA_DIR, "top_drifted_words_unfiltered.json")


def load_year(year):
    vectors = np.load(os.path.join(MODELS_DIR, f"aligned_{year}.npy"))
    with open(os.path.join(MODELS_DIR, f"vocab_{year}.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vectors, vocab


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms


def cosine_distance(vec_a, vec_b):
    return 1.0 - float(np.dot(vec_a, vec_b))


def is_meaningful_word(word, cumulative_drift):
    """Filters out noisy tokens (subreddit jargon, abbreviations, HTML
    artifacts) that tend to dominate a raw drift ranking with sampling noise
    rather than genuine semantic shift."""
    if len(word) < MIN_WORD_LENGTH:
        return False
    if word.isupper():
        return False
    if word in ENGLISH_WORDS:
        return True
    # Not a dictionary word -- still allow it through if it shows a large,
    # sustained drift, since that's the signature of a genuinely emerging
    # word (e.g. "covid", "crypto") rather than noise.
    return cumulative_drift > MIN_DRIFT_FOR_OOV and len(word) >= MIN_LENGTH_FOR_OOV


def compute_neighbors(shared_vocab, norm_vectors, vocab, word_to_idx):
    """Top-K nearest neighbors (by cosine similarity) for each shared word,
    searched against this year's full vocabulary, in batches to bound memory."""
    query_indices = [word_to_idx[w] for w in shared_vocab]
    neighbors = {}

    for start in range(0, len(query_indices), NEIGHBOR_BATCH):
        batch_idx = query_indices[start:start + NEIGHBOR_BATCH]
        batch_words = shared_vocab[start:start + NEIGHBOR_BATCH]
        sims = norm_vectors[batch_idx] @ norm_vectors.T  # [batch, V]

        for i, word in enumerate(batch_words):
            row = sims[i].copy()
            row[batch_idx[i]] = -np.inf  # exclude the word itself
            top_idx = np.argpartition(row, -TOP_K_NEIGHBORS)[-TOP_K_NEIGHBORS:]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            neighbors[word] = [[vocab[j], float(row[j])] for j in top_idx]

    return neighbors


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DRIFT_RESULTS_PATH):
        print(f"{DRIFT_RESULTS_PATH} already exists, loading cached results")
        with open(DRIFT_RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        build_rankings(results)
        return

    print("Loading aligned embeddings...")
    year_data = {}
    for year in YEARS:
        vectors, vocab = load_year(year)
        year_data[year] = {
            "norm_vectors": normalize(vectors),
            "vocab": vocab,
            "word_to_idx": {w: i for i, w in enumerate(vocab)},
        }

    shared_vocab = set(year_data[YEARS[0]]["vocab"])
    for year in YEARS[1:]:
        shared_vocab &= set(year_data[year]["vocab"])
    shared_vocab = sorted(shared_vocab)
    print(f"Shared vocabulary across all {len(YEARS)} years: {len(shared_vocab):,} words")

    print("Computing nearest neighbors per year...")
    neighbors_by_year = {}
    for year in YEARS:
        data = year_data[year]
        neighbors_by_year[year] = compute_neighbors(
            shared_vocab, data["norm_vectors"], data["vocab"], data["word_to_idx"]
        )
        print(f"  {year}: done")

    print("Computing drift...")
    anchor_year = YEARS[0]
    results = {}
    for word in shared_vocab:
        yearly_drift = {}
        for t, t_next in zip(YEARS[:-1], YEARS[1:]):
            vec_t = year_data[t]["norm_vectors"][year_data[t]["word_to_idx"][word]]
            vec_t_next = year_data[t_next]["norm_vectors"][year_data[t_next]["word_to_idx"][word]]
            yearly_drift[f"{t}-{t_next}"] = cosine_distance(vec_t, vec_t_next)

        anchor_vec = year_data[anchor_year]["norm_vectors"][year_data[anchor_year]["word_to_idx"][word]]
        cumulative_drift = {}
        for year in YEARS:
            vec = year_data[year]["norm_vectors"][year_data[year]["word_to_idx"][word]]
            cumulative_drift[str(year)] = cosine_distance(anchor_vec, vec)

        results[word] = {
            "yearly_drift": yearly_drift,
            "cumulative_drift": cumulative_drift,
            "neighbors": {str(year): neighbors_by_year[year][word] for year in YEARS},
        }

    print(f"Saving {DRIFT_RESULTS_PATH}...")
    with open(DRIFT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f)

    build_rankings(results)


def build_rankings(results):
    final_year = YEARS[-1]
    ranked = sorted(
        results.items(),
        key=lambda kv: kv[1]["cumulative_drift"][str(final_year)],
        reverse=True,
    )

    top_drifted_unfiltered = [
        [word, data["cumulative_drift"][str(final_year)]] for word, data in ranked[:TOP_N_DRIFTED]
    ]
    print(f"Saving {TOP_DRIFTED_UNFILTERED_PATH}...")
    with open(TOP_DRIFTED_UNFILTERED_PATH, "w", encoding="utf-8") as f:
        json.dump(top_drifted_unfiltered, f, indent=2)

    filtered_ranked = [
        (word, data) for word, data in ranked
        if is_meaningful_word(word, data["cumulative_drift"][str(final_year)])
    ]
    top_drifted = [
        [word, data["cumulative_drift"][str(final_year)]] for word, data in filtered_ranked[:TOP_N_DRIFTED]
    ]
    print(f"Saving {TOP_DRIFTED_PATH}...")
    with open(TOP_DRIFTED_PATH, "w", encoding="utf-8") as f:
        json.dump(top_drifted, f, indent=2)

    print(f"\nTop 20 most drifted words (filtered, {YEARS[0]} -> {final_year}):")
    for word, drift in top_drifted[:20]:
        print(f"  {word}: {drift:.4f}")


if __name__ == "__main__":
    main()
