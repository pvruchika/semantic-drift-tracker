"""Inspect word-frequency cutoffs for the 2015 corpus to sanity-check min_count."""

from collections import Counter

CORPUS_PATH = "data/processed/reddit_2015.txt"
THRESHOLDS = [5, 10, 20, 30, 50, 100]
EXACT_COUNT = 50
NUM_EXAMPLES = 10
NUM_COMMON = 20


def main():
    counts = Counter()
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            counts.update(line.split())

    print(f"Total unique words: {len(counts):,}\n")

    print("Vocabulary size at frequency cutoffs:")
    for n in THRESHOLDS:
        kept = sum(1 for c in counts.values() if c >= n)
        print(f"  >= {n}: {kept:,} words")

    print(f"\nTop {NUM_COMMON} most common words:")
    for word, count in counts.most_common(NUM_COMMON):
        print(f"  {word}: {count:,}")

    exact = [word for word, count in counts.items() if count == EXACT_COUNT]
    print(f"\n{NUM_EXAMPLES} example words with exactly {EXACT_COUNT} occurrences:")
    for word in exact[:NUM_EXAMPLES]:
        print(f"  {word}")


if __name__ == "__main__":
    main()
