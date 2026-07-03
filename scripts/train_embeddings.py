"""Train a per-year Word2Vec model on the cleaned Reddit corpora."""

import os

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

VECTOR_SIZE = 200
WINDOW = 5
MIN_COUNT = 15
WORKERS = 4
EPOCHS = 5
SG = 1

YEARS = range(2015, 2024)
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"


class LineSentences:
    """Stream whitespace-tokenized lines from a file without loading it all into memory."""

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.split()


class EpochProgressBar(CallbackAny2Vec):
    def __init__(self, year, total_epochs):
        self.pbar = tqdm(total=total_epochs, desc=f"{year}", unit="epoch")

    def on_epoch_end(self, model):
        self.pbar.update(1)

    def on_train_end(self, model):
        self.pbar.close()


def train_year(year):
    model_path = os.path.join(MODELS_DIR, f"w2v_{year}.model")

    if os.path.exists(model_path):
        print(f"{year}: model already exists, skipping")
        model = Word2Vec.load(model_path)
        return len(model.wv)

    corpus_path = os.path.join(PROCESSED_DIR, f"reddit_{year}.txt")
    sentences = LineSentences(corpus_path)
    callback = EpochProgressBar(year, EPOCHS)

    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        epochs=EPOCHS,
        sg=SG,
        callbacks=[callback],
    )

    model.save(model_path)

    vocab_size = len(model.wv)
    print(f"{year}: vocabulary size = {vocab_size:,}")
    return vocab_size


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    vocab_sizes = {}
    for year in YEARS:
        vocab_sizes[year] = train_year(year)

    print("\nSummary:")
    print(f"{'Year':<6}{'Vocab size':>12}")
    for year, size in vocab_sizes.items():
        print(f"{year:<6}{size:>12,}")


if __name__ == "__main__":
    main()
