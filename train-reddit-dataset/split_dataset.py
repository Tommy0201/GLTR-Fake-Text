"""
Split a CSV dataset into two equal halves.

Usage:
    python3 split_dataset.py --input reddit_dataset_combined.csv
    python3 split_dataset.py --input raw/reddit_dataset_10k.csv \
        --first first_half.csv --second second_half.csv
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reddit_dataset_combined.csv")
    parser.add_argument("--first", default=None, help="Output path for first half")
    parser.add_argument("--second", default=None, help="Output path for second half")
    args = parser.parse_args()

    input_path = Path(args.input)
    first_path = Path(args.first) if args.first else input_path.with_name(input_path.stem + "_first_half.csv")
    second_path = Path(args.second) if args.second else input_path.with_name(input_path.stem + "_second_half.csv")

    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    mid = len(rows) // 2

    for path, data in ((first_path, rows[:mid]), (second_path, rows[mid:])):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Wrote {len(data)} rows -> {path}")


if __name__ == "__main__":
    main()
