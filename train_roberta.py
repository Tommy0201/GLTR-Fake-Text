"""Fine-tune RoBERTa for binary classification (human vs AI-generated text) using text datasets.

Supports single file or combining multiple files for training scenarios like odd-one-out.
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (RobertaForSequenceClassification, RobertaTokenizer,
                          Trainer, TrainingArguments)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_csv_data(path: Path):
    """Load preprocessed CSV with columns: label, text, source."""
    texts = []
    labels = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label = row.get("label", "").strip()
            
            if text and label:
                texts.append(text)
                labels.append(int(label))

    if not texts:
        raise ValueError(f"No valid text data found in {path}")

    return texts, labels


def combine_csv_files(file_paths):
    """Combine multiple CSV files into a single dataset."""
    all_texts = []
    all_labels = []
    
    for file_path in file_paths:
        texts, labels = load_csv_data(file_path)
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"Loaded {len(texts)} samples from {file_path.name}")
    
    print(f"Combined total: {len(all_texts)} samples")
    return all_texts, all_labels


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on text classification datasets")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("train-dataset-text/Essay_LLMs_processed.csv"),
        help="Path to preprocessed training data CSV",
    )
    parser.add_argument(
        "--train-files",
        nargs="+",
        type=Path,
        help="List of preprocessed training data CSV files to combine (overrides --train-file)",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Path to preprocessed test data CSV (if not provided, splits train-file into 80-20)",
    )
    parser.add_argument(
        "--test-files",
        nargs="+",
        type=Path,
        help="List of preprocessed test data CSV files to combine (overrides --test-file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/roberta_model"),
        help="Path to save trained model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="roberta-base",
        help="Pre-trained RoBERTa model name",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lr-scheduler-type", type=str, default="linear", 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="Learning rate scheduler type")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.random_state)

    # Load training data
    if args.train_files:
        # Combine multiple training files
        train_texts, train_labels = combine_csv_files(args.train_files)
    else:
        # Use single training file
        train_texts, train_labels = load_csv_data(args.train_file)

    if args.test_files:
        # Combine multiple test files
        test_texts, test_labels = combine_csv_files(args.test_files)
    elif args.test_file:
        # Use separate test file
        test_texts, test_labels = load_csv_data(args.test_file)
    else:
        # Split train data into train/test
        from sklearn.model_selection import train_test_split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, stratify=train_labels, random_state=args.random_state
        )

    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, args.max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, args.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=str(args.output / "logs"),
        logging_steps=10,
        seed=args.random_state,
    )

    # Define compute_metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate()
    print(f"Test results: {results}")

    # Save model
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
    
    
    
""" TRAINING COMMAND
All Mixed
python train_roberta.py \      
  --train-files train-dataset-text/{cheat, hc3, Essay_LLMs, Reuters_LLMs, WP_LLMs}_processed.csv \
  --output models/roberta/roberta_mixed \
  --epochs 1 --lr-scheduler-type cosine --warmup-steps 200



No MGTBench
python train_roberta.py \
  --train-files train-dataset-text/cheat_processed.csv train-dataset-text/hc3_processed.csv \
  --test-files train-dataset-text/Essay_LLMs_processed.csv train-dataset-text/Reuters_LLMs_processed.csv train-dataset-text/WP_LLMs_processed.csv \
  --output models/roberta/roberta_no_mgt \
  --epochs 1 --lr-scheduler-type cosine --warmup-steps 200
"""