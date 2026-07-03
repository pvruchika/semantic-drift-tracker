"""Fetch Reddit comments per year from the Arctic Shift API and write them
to data/raw/comments_{year}.txt."""

import calendar
import datetime
import os
import time

import requests

API_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"
YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
RAW_DIR = "data/raw"
PAGE_LIMIT = 100
TARGET_PER_YEAR = 300_000
MIN_BODY_LENGTH = 20
REQUEST_DELAY = 0.3
RETRY_DELAY = 2.0
PROGRESS_INTERVAL = 5_000


def fetch_page(after_ts, before_ts):
    params = {
        "after": after_ts,
        "before": before_ts,
        "limit": PAGE_LIMIT,
        "sort": "asc",
    }
    try:
        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        time.sleep(RETRY_DELAY)
        try:
            response = requests.get(API_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed after retry, skipping page: {e}")
            return None


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def download_year(year):
    out_path = os.path.join(RAW_DIR, f"comments_{year}.txt")
    existing = count_lines(out_path)
    if existing >= TARGET_PER_YEAR:
        print(f"Skipping {year}: already complete ({existing:,} comments) -> {out_path}")
        return existing

    after_ts = calendar.timegm(datetime.date(year, 1, 1).timetuple())
    before_ts = calendar.timegm(datetime.date(year + 1, 1, 1).timetuple())

    collected = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        while collected < TARGET_PER_YEAR:
            payload = fetch_page(after_ts, before_ts)
            time.sleep(REQUEST_DELAY)

            if payload is None:
                break

            comments = payload.get("data", payload) if isinstance(payload, dict) else payload
            if not comments:
                break

            for comment in comments:
                body = comment.get("body")
                if body is None or body in ("[deleted]", "[removed]"):
                    continue
                if len(body) < MIN_BODY_LENGTH:
                    continue

                out_f.write(body.replace("\n", " ").replace("\r", " ") + "\n")
                collected += 1

                if collected % PROGRESS_INTERVAL == 0:
                    print(f"{year}: collected {collected:,} comments")

                if collected >= TARGET_PER_YEAR:
                    break

            after_ts = comments[-1]["created_utc"] + 1

            if len(comments) < PAGE_LIMIT:
                break

    print(f"Finished {year}: {collected:,} comments -> {out_path}")
    return collected


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    final_counts = {}
    for year in YEARS:
        final_counts[year] = download_year(year)

    print("Final counts per year:")
    for year, count in final_counts.items():
        print(f"  {year}: {count:,}")


if __name__ == "__main__":
    main()
