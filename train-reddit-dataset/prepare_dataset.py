"""
Prepare a labeled dataset from Reddit JSONL files.

All Reddit posts are human-written (label=0). This script extracts the
relevant fields and writes a CSV ready for fake-text detection training.

Usage:
    python prepare_dataset.py file1.jsonl file2.jsonl file3.jsonl [-o output.csv]
"""

import argparse
import csv
import json
import os
import sys


MIN_TEXT_LENGTH = 50


def extract_text(record: dict) -> str:
    """Return the best text content: selftext if substantive, else title."""
    selftext = (record.get("selftext") or "").strip()
    title = (record.get("title") or "").strip()
    if len(selftext) >= MIN_TEXT_LENGTH:
        return selftext
    if selftext:
        return f"{title} {selftext}".strip()
    return title


def process_file(path: str) -> list[dict]:
    rows = []
    source = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {source}:{lineno} — JSON parse error: {e}", file=sys.stderr)
                continue

            text = extract_text(record)
            if len(text) < MIN_TEXT_LENGTH:
                continue

            rows.append({
                "id": record.get("id", ""),
                "text": text,
                "label": 0,
                "source_file": source,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare Reddit dataset for fake-text detection.")
    parser.add_argument("files", nargs=3, metavar="FILE", help="Three JSONL input files")
    parser.add_argument("-o", "--output", default="reddit_dataset.csv", help="Output CSV path (default: reddit_dataset.csv)")
    args = parser.parse_args()

    all_rows = []
    for path in args.files:
        if not os.path.isfile(path):
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            sys.exit(1)
        rows = process_file(path)
        print(f"{os.path.basename(path)}: {len(rows)} records extracted")
        all_rows.extend(rows)

    fieldnames = ["id", "text", "label", "source_file"]
    output_path = args.output
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nTotal records: {len(all_rows)}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
