import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    f1_score,
)

# GPT generated

def main(csv_path):
    # Load CSV (no assumptions about headers)
    df = pd.read_csv(csv_path, header=None, sep=';')

    if df.shape[1] < 2:
        raise ValueError("CSV must have at least two columns (true, predicted)")

    y_true = df.iloc[:, 0]
    y_pred = df.iloc[:, 1]

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=sorted(y_true.unique()), zero_division=0
    )

    # Macro & micro F1
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # Report
    print("=" * 60)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Micro F1: {f1_micro:.4f}")
    print("=" * 60)
    print()

    print("Per-class metrics:")
    print("-" * 60)
    print(f"{'Class':20s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Supp':>8s}")
    print("-" * 60)

    for cls, p, r, f, s in zip(sorted(y_true.unique()), precision, recall, f1, support):
        print(f"{str(cls):20s} {p:8.4f} {r:8.4f} {f:8.4f} {s:8d}")

    print("-" * 60)
    print()

    # Optional sklearn-style report
    print("Full classification report:")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <predictions.csv>")
        sys.exit(1)

    main(sys.argv[1])
