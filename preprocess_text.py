"""Preprocess MGTBench, CHEAT, and HC3 datasets into training-ready format for RoBERTa fine-tuning.

Converts datasets into CSV format with columns: label, text, source
- label: 0 for human, 1 for AI-generated
- text: The text content
- source: Source model name or "human"

Supported datasets:
- MGTBench (local CSV files from eval/mgtbench/)
- CHEAT (from HuggingFace, 5k rows default)
- HC3 (from HuggingFace, 3k rows default)

Usage:
    python preprocess_text.py
"""

import csv
from pathlib import Path


def preprocess_mgtbench_file(input_file: Path, output_file: Path):
    """
    Process a single MGTBench CSV file and save in training format.
    
    Assumes input CSV has columns: id, prompt, human, and AI model columns
    """
    rows = []
    
    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        if fieldnames is None:
            raise ValueError(f"File has no header: {input_file}")
        
        # AI columns: all columns except metadata (id, author_id, author, prompt, human)
        ai_columns = [col for col in fieldnames if col not in ["id", "author_id", "author", "prompt", "human"]]
        
        for row in reader:
            # Process human text (label 0)
            human_text = row.get("human", "").strip()
            if human_text:
                rows.append({
                    "label": 0,
                    "text": human_text,
                    "source": "human"
                })
            
            # Process AI text (label 1)
            for ai_col in ai_columns:
                ai_text = row.get(ai_col, "").strip()
                if ai_text:
                    rows.append({
                        "label": 1,
                        "text": ai_text,
                        "source": ai_col  # e.g., ChatGPT-turbo, Claude, etc.
                    })
    
    if not rows:
        print(f"Warning: No valid text found in {input_file}")
        return
    
    # Shuffle the dataset
    import random
    random.shuffle(rows)
    
    # Save to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "text", "source"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Processed and shuffled {len(rows)} rows: {input_file.name} → {output_file.name}")


def load_and_preprocess_cheat(limit=5000):
    """
    Load CHEAT dataset from HuggingFace and preprocess into training format.
    
    CHEAT columns: text, model (null=human)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets library not installed. Skipping CHEAT dataset.")
        print("Install with: pip install datasets")
        return None
    
    print(f"Downloading acmc/cheat (limiting to {limit} rows)...")
    try:
        ds = load_dataset("acmc/cheat", split="train")
    except Exception as e:
        print(f"Error loading CHEAT dataset: {e}")
        return None
    
    human_rows, ai_rows = [], []
    for i, row in enumerate(ds):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        ai_model = (row.get("model") or "").strip()
        
        if ai_model:  # AI-generated
            ai_rows.append({
                "label": 1,
                "text": text,
                "source": ai_model
            })
        else:  # Human
            human_rows.append({
                "label": 0,
                "text": text,
                "source": "human"
            })
    
    # Sample evenly: ~2.5k human, ~2.5k AI
    half = limit // 2
    human_rows = human_rows[:half]
    ai_rows = ai_rows[:limit - half]
    
    result = human_rows + ai_rows
    
    # Shuffle the combined dataset
    import random
    random.shuffle(result)
    
    print(f"Loaded and shuffled {len(result)} samples from CHEAT — {len(human_rows)} human, {len(ai_rows)} AI")
    return result


def load_and_preprocess_hc3(limit=3000):
    """
    Load HC3 dataset from HuggingFace and preprocess into training format.
    
    HC3 structure: each row has question, human_answers list, chatgpt_answers list
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets library not installed. Skipping HC3 dataset.")
        print("Install with: pip install datasets")
        return None
    
    print(f"Downloading Hello-SimpleAI/HC3 (limiting to {limit} rows)...")
    try:
        ds = load_dataset(
            "json",
            data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl",
            split="train",
        )
    except Exception as e:
        print(f"Error loading HC3 dataset: {e}")
        return None
    
    human_rows, ai_rows = [], []
    for i, row in enumerate(ds):
        # Extract human answers
        for j, text in enumerate((row.get("human_answers") or [])):
            text = text.strip()
            if text:
                human_rows.append({
                    "label": 0,
                    "text": text,
                    "source": "human"
                })
        
        # Extract ChatGPT answers
        for j, text in enumerate((row.get("chatgpt_answers") or [])):
            text = text.strip()
            if text:
                ai_rows.append({
                    "label": 1,
                    "text": text,
                    "source": "ChatGPT"
                })
    
    # Sample evenly: ~1.5k human, ~1.5k AI
    half = limit // 2
    human_rows = human_rows[:half]
    ai_rows = ai_rows[:limit - half]
    
    result = human_rows + ai_rows
    
    # Shuffle the combined dataset
    import random
    random.shuffle(result)
    
    print(f"Loaded and shuffled {len(result)} samples from HC3 — {len(human_rows)} human, {len(ai_rows)} AI")
    return result


def save_preprocessed_data(rows, output_file: Path, dataset_name: str):
    """Save preprocessed rows to CSV file."""
    if not rows:
        print(f"Warning: No rows to save for {dataset_name}")
        return
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "text", "source"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Saved {len(rows)} rows to {output_file.name}")



def main():
    mgtbench_dir = Path("eval/mgtbench")
    output_dir = Path("train-dataset-text")
    
    # Process MGTBench files
    print("=" * 70)
    print("Processing MGTBench files...")
    print("=" * 70)
    if mgtbench_dir.exists():
        for csv_file in mgtbench_dir.glob("*.csv"):
            output_file = output_dir / f"{csv_file.stem}_processed.csv"
            preprocess_mgtbench_file(csv_file, output_file)
    else:
        print(f"Warning: {mgtbench_dir} not found, skipping MGTBench")
    
    # Process CHEAT dataset from HuggingFace
    print("\n" + "=" * 70)
    print("Processing CHEAT dataset...")
    print("=" * 70)
    cheat_rows = load_and_preprocess_cheat(limit=5000)
    if cheat_rows:
        cheat_output = output_dir / "cheat_processed.csv"
        save_preprocessed_data(cheat_rows, cheat_output, "CHEAT")
    
    # Process HC3 dataset from HuggingFace
    print("\n" + "=" * 70)
    print("Processing HC3 dataset...")
    print("=" * 70)
    hc3_rows = load_and_preprocess_hc3(limit=3000)
    if hc3_rows:
        hc3_output = output_dir / "hc3_processed.csv"
        save_preprocessed_data(hc3_rows, hc3_output, "HC3")
    
    print("\n" + "=" * 70)
    print("Preprocessing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
