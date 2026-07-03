"""Align per-year Word2Vec embeddings into a shared coordinate space.

Each year's Word2Vec model is trained independently, starting from random
initialization, so its vector space ends up as an arbitrary rotation/reflection
relative to any other year's space. The geometry *within* a year is meaningful
(similar words cluster together), but a word's coordinates in 2015 and the same
word's coordinates in 2020 are not directly comparable -- the two spaces are
unrelated coordinate systems. Computing "semantic drift" as e.g. a vector
difference or cosine similarity between the same word across years would
therefore be meaningless without first rotating one space onto the other.

Orthogonal Procrustes finds the rotation matrix that best maps one year's
vectors onto the anchor year's vectors, using only the vocabulary the two years
share. Because the mapping is restricted to be orthogonal (a pure
rotation/reflection, no scaling or shearing), it preserves each space's
internal geometry -- distances and angles between words don't change -- while
making cross-year comparisons valid.
"""

import json
import os

import numpy as np
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes

ANCHOR_YEAR = 2015
YEARS = range(2015, 2024)
MODELS_DIR = "models"
TOP_N_EVAL = 100


def load_model(year):
    return Word2Vec.load(os.path.join(MODELS_DIR, f"w2v_{year}.model"))


def save_aligned(year, vectors, vocab):
    np.save(os.path.join(MODELS_DIR, f"aligned_{year}.npy"), vectors)
    with open(os.path.join(MODELS_DIR, f"vocab_{year}.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def eval_alignment_quality(anchor_model, year_vocab, vectors_before, vectors_after):
    """Average cosine similarity of the anchor's top-N most common words
    against the year's vectors, before vs. after rotation."""
    year_index = {word: i for i, word in enumerate(year_vocab)}
    top_words = [w for w in anchor_model.wv.index_to_key[:TOP_N_EVAL] if w in year_index]

    before_sims = []
    after_sims = []
    for word in top_words:
        anchor_vec = anchor_model.wv[word]
        idx = year_index[word]
        before_sims.append(cosine_sim(vectors_before[idx], anchor_vec))
        after_sims.append(cosine_sim(vectors_after[idx], anchor_vec))

    return np.mean(before_sims), np.mean(after_sims), len(top_words)


def align_year(year, anchor_model):
    year_model = load_model(year)
    year_vocab = year_model.wv.index_to_key
    year_vectors = year_model.wv.vectors

    shared_words = sorted(set(year_vocab) & set(anchor_model.wv.index_to_key))
    A = np.array([year_model.wv[w] for w in shared_words])
    B = np.array([anchor_model.wv[w] for w in shared_words])

    # R is the orthogonal (rotation/reflection) matrix minimizing ||A @ R - B||,
    # i.e. the best rigid mapping of this year's shared-vocab vectors onto the
    # anchor's. Applying it to the *full* vocabulary carries that same rotation
    # over to every word, not just the ones used to compute it.
    R, _ = orthogonal_procrustes(A, B)
    aligned_vectors = year_vectors @ R

    before, after, n_eval = eval_alignment_quality(anchor_model, year_vocab, year_vectors, aligned_vectors)
    print(
        f"{year}: shared vocab = {len(shared_words):,} words | "
        f"top-{n_eval} common-word cosine sim: before={before:.4f} after={after:.4f}"
    )

    return aligned_vectors, year_vocab


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    anchor_model = load_model(ANCHOR_YEAR)
    anchor_path = os.path.join(MODELS_DIR, f"aligned_{ANCHOR_YEAR}.npy")

    if os.path.exists(anchor_path):
        print(f"{ANCHOR_YEAR}: already aligned, skipping")
    else:
        save_aligned(ANCHOR_YEAR, anchor_model.wv.vectors, anchor_model.wv.index_to_key)
        print(f"{ANCHOR_YEAR}: saved as anchor (reference space, no rotation needed)")

    for year in YEARS:
        if year == ANCHOR_YEAR:
            continue

        out_path = os.path.join(MODELS_DIR, f"aligned_{year}.npy")
        if os.path.exists(out_path):
            print(f"{year}: already aligned, skipping")
            continue

        aligned_vectors, year_vocab = align_year(year, anchor_model)
        save_aligned(year, aligned_vectors, year_vocab)

    print("Done.")


if __name__ == "__main__":
    main()
