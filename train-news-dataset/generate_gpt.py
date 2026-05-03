"""
Generate AI-rewritten versions of Reddit posts using the OpenAI API,
then combine with the original human-written CSV for binary classification.

Labels:  0 = human (Reddit)   1 = AI (GPT-rewritten)

Usage:
    export OPENAI_API_KEY=sk-...

    # Generate AI dataset and NOT COMBINE
    python3 generate_gpt.py --input  articles_content_10k.csv --output gpt/gpt_dataset.csv
        
    # Generate AI dataset and combine:
    python3 generate_gpt.py \
        --input  reddit_dataset_combined.csv \
        --output gpt_dataset.csv \
        --combined combined_dataset.csv \
        --limit 10

    # Dry-run first 20 rows to check cost/output:
    python generate_gpt.py --input reddit_dataset_combined.csv --limit 20

Checkpointing:
    Progress is saved to <output>.checkpoint.json after every batch.
    Re-running the same command resumes from where it left off.
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

import openai

MODEL = "gpt-5.4"
CONCURRENCY = 5          # simultaneous API requests
BATCH_SAVE_EVERY = 100   # write checkpoint after this many completions

SYSTEM_PROMPT = (
    "You are an expert in paraphrasing human text. Rewrite the following text in your own tone "
    "and style, preserving the meaning but making it sound naturally written by you. "
    "Return only the rewritten text — no preamble, no labels, no quotes."
)


def load_checkpoint(path: str) -> dict[str, str]:
    """Return {id: generated_text} for already-completed rows."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(path: str, done: dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False)


def read_human_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def rewrite_one(
    client: openai.AsyncOpenAI,
    row: dict,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> tuple[str, str | None]:
    """Return (id, generated_text | None on failure)."""
    text = row.get("content") or row.get("text", "")
    for attempt in range(retries):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.9,
                )
            result = row["id"], resp.choices[0].message.content.strip()
            await asyncio.sleep(0.5)
            return result
        except openai.RateLimitError as e:
            wait = 2 ** attempt * 5
            print(f"  [rate limit] id={row['id']}: {e}", flush=True)
            print(f"  waiting {wait}s …", flush=True)
            await asyncio.sleep(wait)
        except openai.AuthenticationError as e:
            print(f"  [auth error] check your OPENAI_API_KEY: {e}", flush=True)
            return row["id"], None
        except openai.APIError as e:
            print(f"  [api error] id={row['id']}: {e}", flush=True)
            await asyncio.sleep(2)
    return row["id"], None


async def generate_all(
    rows: list[dict],
    checkpoint: dict[str, str],
    checkpoint_path: str,
) -> dict[str, str]:
    pending = [r for r in rows if r["id"] not in checkpoint]
    print(f"  {len(checkpoint)} already done, {len(pending)} remaining")

    if not pending:
        return checkpoint

    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    semaphore = asyncio.Semaphore(CONCURRENCY)
    done = dict(checkpoint)
    tasks = [rewrite_one(client, r, semaphore) for r in pending]

    completed = 0
    start = time.time()
    for coro in asyncio.as_completed(tasks):
        row_id, text = await coro
        if text is not None:
            done[row_id] = text
        else:
            print(f"  [SKIP] id={row_id} — all retries failed", flush=True)

        completed += 1
        if completed % 50 == 0:
            elapsed = time.time() - start
            rate = completed / elapsed
            remaining = (len(pending) - completed) / rate if rate > 0 else 0
            print(
                f"  {completed}/{len(pending)}  "
                f"({rate:.1f}/s  ~{remaining/60:.1f} min left)",
                flush=True,
            )

        if completed % BATCH_SAVE_EVERY == 0:
            save_checkpoint(checkpoint_path, done)
            print(f"  [checkpoint] saved {len(done)} entries", flush=True)

    save_checkpoint(checkpoint_path, done)
    return done


def write_gpt_csv(rows: list[dict], generated: dict[str, str], out_path: str) -> int:
    fieldnames = ["id", "article_id", "title", "content", "label"]
    written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            gpt_text = generated.get(row["id"])
            if not gpt_text:
                continue
            writer.writerow({
                "id": row["id"],
                "article_id": row.get("article_id", ""),
                "title": row.get("title", ""),
                "content": gpt_text,
                "label": 1,
            })
            written += 1
    return written


def write_combined_csv(human_path: str, gpt_path: str, out_path: str) -> int:
    fieldnames = ["id", "article_id", "title", "content", "label"]
    rows = []
    with open(human_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": r["id"],
                "article_id": r.get("article_id", ""),
                "title": r.get("title", ""),
                "content": r.get("content") or r.get("text", ""),
                "label": r.get("label", 0),
            })
    with open(gpt_path, newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reddit_dataset_combined.csv")
    parser.add_argument("--output", default="gpt_dataset.csv")
    parser.add_argument("--combined", default="combined_dataset.csv",
                        help="Path for the merged human+AI CSV (set to '' to skip)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows (for testing)")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    MODEL = args.model

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    rows = read_human_csv(input_path)
    if args.limit:
        rows = rows[: args.limit]

    checkpoint_path = args.output + ".checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)

    print(f"Input:  {input_path}  ({len(rows)} rows)")
    print(f"Model:  {MODEL}")
    print(f"Output: {args.output}")

    generated = asyncio.run(generate_all(rows, checkpoint, checkpoint_path))

    n_written = write_gpt_csv(rows, generated, args.output)
    print(f"\nGPT dataset: {n_written} rows written to {args.output}")

    if args.combined:
        n_combined = write_combined_csv(input_path, args.output, args.combined)
        print(f"Combined dataset: {n_combined} rows written to {args.combined}")
        label_counts = {}
        with open(args.combined, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        print(f"  label=0 (human): {label_counts.get('0', 0)}")
        print(f"  label=1 (AI):    {label_counts.get('1', 0)}")


if __name__ == "__main__":
    main()
