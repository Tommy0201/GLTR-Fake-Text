"""
Sample 10k high-quality records from each clean_RS_*.jsonl file.

Quality criteria:
  - score >= 5  (some upvotes)
  - text length >= 100 chars
  - selftext not deleted/removed
  - text derived from selftext (not title-only)

Uses reservoir sampling (Algorithm R) so only O(k) memory is needed
regardless of how many records pass the quality filter.

Usage:
    python sample_high_quality.py [--n 10000] [--seed 42] [--out-dir .]
"""

import argparse
import json
import os
import random
import sys
from glob import glob


MIN_TEXT_LEN = 100
MIN_SCORE = 5
DELETED = {"[deleted]", "[removed]", ""}


def is_high_quality(record: dict) -> bool:
    selftext = (record.get("selftext") or "").strip()
    if selftext in DELETED:
        return False
    if len(selftext) < MIN_TEXT_LEN:
        return False
    if record.get("score", 0) < MIN_SCORE:
        return False
    return True


def extract_text(record: dict) -> str:
    selftext = (record.get("selftext") or "").strip()
    title = (record.get("title") or "").strip()
    if len(selftext) >= MIN_TEXT_LEN:
        return selftext
    return f"{title} {selftext}".strip()


def reservoir_sample(path: str, k: int, rng: random.Random) -> list[dict]:
    """Single-pass reservoir sampling over qualifying lines."""
    reservoir = []
    seen = 0

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not is_high_quality(record):
                continue

            seen += 1
            if len(reservoir) < k:
                reservoir.append(record)
            else:
                j = rng.randint(0, seen - 1)
                if j < k:
                    reservoir[j] = record

            if lineno % 500_000 == 0:
                print(f"  ...{lineno:,} lines scanned, {seen:,} qualifying so far", flush=True)

    return reservoir, seen


def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            out = {
                "id": rec.get("id", ""),
                "text": extract_text(rec),
                "label": 0,
                "subreddit": rec.get("subreddit", ""),
                "score": rec.get("score", 0),
                "num_comments": rec.get("num_comments", 0),
                "created_utc": rec.get("created_utc", 0),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3_333, help="Records to sample per file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=".", help="Directory for output files")
    args = parser.parse_args()

    input_dir = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob(os.path.join(input_dir, "clean_RS_*.jsonl")))

    if not files:
        print("No clean_RS_*.jsonl files found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    for path in files:
        basename = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out_dir, f"sample_{basename}.jsonl")
        print(f"\n{os.path.basename(path)}")

        sample, total_qualifying = reservoir_sample(path, args.n, rng)

        if total_qualifying < args.n:
            print(f"  WARNING: only {total_qualifying} qualifying records, wanted {args.n}")

        write_jsonl(sample, out_path)
        print(f"  {total_qualifying:,} qualifying → sampled {len(sample):,} → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
