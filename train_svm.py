"""Train an SVM classifier from eval/train-dataset-stats/train.csv or separate train/test files."""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        if header is None:
            raise ValueError(f"File has no header: {path}")
        rows = [row for row in reader]

    if not rows:
        raise ValueError(f"No rows found in {path}")

    feature_names = [c for c in header if c != "target"]
    X = np.array([[float(row[col]) for col in feature_names] for row in rows], dtype=float)
    y = np.array([int(row["target"]) for row in rows], dtype=int)
    return X, y, feature_names


def train_svm(X_train, y_train, X_test, y_test, kernel="rbf", C=1.0, gamma="scale", random_state=0):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=random_state),
        ),
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    return pipeline, score


def main():
    parser = argparse.ArgumentParser(description="Train an SVM on train.csv or separate train/test files")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("eval/train-dataset-stats/train4cols.csv"),
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Path to test data CSV (if not provided, splits train-file into 80-20)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/svm_model.pkl"),
        help="Path to save trained model",
    )
    parser.add_argument("--kernel", choices=["linear", "rbf", "poly", "sigmoid"], default="rbf")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    # Load training data
    X_train, y_train, features = load_csv(args.train_file)
    
    if args.test_file:
        # Use separate test file
        X_test, y_test, test_features = load_csv(args.test_file)
        if features != test_features:
            raise ValueError("Train and test files must have the same feature columns")
    else:
        # Split train file into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=args.random_state
        )

    model, score = train_svm(
        X_train, y_train, X_test, y_test, 
        kernel=args.kernel, C=args.C, gamma=args.gamma, random_state=args.random_state
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"model": model, "features": features}, f)

    print(f"Trained SVM model using {len(features)} features: {features}")
    print(f"Saved model to {args.output}")
    print(f"Test accuracy: {score:.4f}")


if __name__ == "__main__":
    main()

"""
Mixed/Combined:
python3 train_svm.py --train-file eval/train-dataset-stats/train4cols.csv --output models/svm_model.pkl

Odd-One-Out:
python3 train_svm.py --train-file eval/train-dataset-stats/scenario1_train.csv --test-file eval/train-dataset-stats/scenario1_test.csv --output models/svm_scenario1.pkl
python3 train_svm.py --train-file eval/train-dataset-stats/scenario2_train.csv --test-file eval/train-dataset-stats/scenario2_test.csv --output models/svm_scenario2.pkl
python3 train_svm.py --train-file eval/train-dataset-stats/scenario3_train.csv --test-file eval/train-dataset-stats/scenario3_test.csv --output models/svm_scenario3.pkl

Odd-One-Out within MGTBench:
python3 train_svm.py --train-file eval/train-dataset-stats/mgtbench_essay_GPT2_train.csv --test-file eval/train-dataset-stats/mgtbench_essay_GPT2_test.csv --output models/svm_mgt_essay.pkl
python3 train_svm.py --train-file eval/train-dataset-stats/mgtbench_reuters_GPT2_train.csv --test-file eval/train-dataset-stats/mgtbench_reuters_GPT2_test.csv --output models/svm_mgt_reuters.pkl
python3 train_svm.py --train-file eval/train-dataset-stats/mgtbench_wp_GPT2_train.csv --test-file eval/train-dataset-stats/mgtbench_wp_GPT2_test.csv --output models/svm_mgt_wp.pkl
"""