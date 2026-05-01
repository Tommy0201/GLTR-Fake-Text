"""
Generate AI-rewritten versions of Reddit posts using the Google Gemini API,
then combine with the original human-written CSV for binary classification.

Labels:  0 = human (Reddit)   1 = AI (Gemini-rewritten)

Usage:
    export GEMINI_API_KEY=...

    # Generate AI dataset and combine:
    
    python3 generate_gemini_dataset.py \
        --input  reddit_first_half.csv \
        --output gemini/gemini_dataset.csv \
        --limit 2
        --combined gemini_combined_dataset.csv

    # Dry-run first 10 rows:
    python3 generate_gemini_dataset.py --input reddit_dataset_combined.csv --limit 10 \
        --output gemini_sample_test.csv --combined combined_sample_test.csv

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

import google.generativeai as genai

MODEL = "gemini-3-flash-preview"
CONCURRENCY = 5        # simultaneous API requests
BATCH_SAVE_EVERY = 100  # write checkpoint after this many completions

SYSTEM_PROMPT = (
    "You are an expert in paraphrasing human text. Rewrite the following text in your own tone "
    "and style, preserving the meaning but making it sound naturally written by you. "
    "Return only the rewritten text — no preamble, no labels, no quotes."
)


def load_checkpoint(path: str) -> dict[str, str]:
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
    model: genai.GenerativeModel,
    row: dict,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> tuple[str, str | None]:
    prompt = f"{SYSTEM_PROMPT}\n\n{row['text']}"
    for attempt in range(retries):
        try:
            async with semaphore:
                resp = await model.generate_content_async(prompt)
            result = row["id"], resp.text.strip()
            await asyncio.sleep(0.5)
            return result
        except Exception as e:
            err = str(e)
            if "404" in err:
                print(f"  [model error] id={row['id']}: {e}", flush=True)
                return row["id"], None
            elif "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = 2 ** attempt * 5
                print(f"  [rate limit] id={row['id']}, waiting {wait}s …", flush=True)
                await asyncio.sleep(wait)
            else:
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

    model = genai.GenerativeModel(MODEL)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    done = dict(checkpoint)
    tasks = [rewrite_one(model, r, semaphore) for r in pending]

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


def write_gemini_csv(rows: list[dict], generated: dict[str, str], out_path: str) -> int:
    fieldnames = ["id", "text", "label", "source_file"]
    written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            gemini_text = generated.get(row["id"])
            if not gemini_text:
                continue
            writer.writerow({
                "id": row["id"],
                "text": gemini_text,
                "label": 1,
                "source_file": row.get("source_file", ""),
            })
            written += 1
    return written


def write_combined_csv(human_path: str, ai_path: str, out_path: str) -> int:
    fieldnames = ["id", "text", "label", "source_file"]
    rows = []
    for path in (human_path, ai_path):
        with open(path, newline="", encoding="utf-8") as f:
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
    parser.add_argument("--output", default="gemini_dataset.csv")
    parser.add_argument("--combined", default="combined_dataset.csv",
                        help="Path for the merged human+AI CSV (set to '' to skip)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows (for testing)")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=api_key)

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

    n_written = write_gemini_csv(rows, generated, args.output)
    print(f"\nGemini dataset: {n_written} rows written to {args.output}")

    if args.combined:
        n_combined = write_combined_csv(input_path, args.output, args.combined)
        print(f"Combined dataset: {n_combined} rows written to {args.combined}")
        label_counts = {}
        with open(args.combined, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        print(f"  label=0 (human):  {label_counts.get('0', 0)}")
        print(f"  label=1 (Gemini): {label_counts.get('1', 0)}")


if __name__ == "__main__":
    main()
