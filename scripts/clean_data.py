"""Clean raw Reddit comment text files into tokenized per-year corpora."""

import os
import re

YEARS = range(2015, 2024)
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MIN_TOKENS = 5

URL_RE = re.compile(r"https?://\S+|www\.\S+")
APOSTROPHE_RE = re.compile(r"'")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_line(line):
    line = line.lower()
    line = URL_RE.sub(" ", line)
    line = APOSTROPHE_RE.sub("", line)
    line = NON_ALPHA_RE.sub(" ", line)
    tokens = line.split()
    return [t for t in tokens if len(t) > 1]


def process_year(year):
    raw_path = os.path.join(RAW_DIR, f"comments_{year}.txt")
    if not os.path.exists(raw_path):
        print(f"Skipping {year}: {raw_path} not found")
        return

    out_path = os.path.join(PROCESSED_DIR, f"reddit_{year}.txt")
    total_tokens = 0

    with open(raw_path, "r", encoding="utf-8") as in_f, \
            open(out_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            tokens = clean_line(line)
            if len(tokens) < MIN_TOKENS:
                continue
            out_f.write(" ".join(tokens) + "\n")
            total_tokens += len(tokens)

    print(f"{year}: {total_tokens:,} tokens -> {out_path}")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for year in YEARS:
        process_year(year)


if __name__ == "__main__":
    main()
