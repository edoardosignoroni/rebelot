import argparse
import json
import os
from datasets import load_dataset, load_from_disk

# GPT generated

def main():
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face dataset to JSONL format."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Hugging Face dataset name (e.g. 'imdb') or local dataset path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to export (default: train)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (default: <dataset>_<split>.jsonl)"
    )

    args = parser.parse_args()

    # Determine if local dataset or HF hub dataset
    if os.path.exists(args.dataset_path):
        print(f"Loading local dataset from disk: {args.dataset_path}")
        dataset = load_from_disk(args.dataset_path)
    else:
        print(f"Loading dataset from Hugging Face hub: {args.dataset_path}")
        dataset = load_dataset(args.dataset_path)

    # If dataset has splits, select requested split
    if isinstance(dataset, dict):
        if args.split not in dataset:
            raise ValueError(
                f"Split '{args.split}' not found. Available splits: {list(dataset.keys())}"
            )
        dataset = dataset[args.split]

    # Define output file
    if args.output:
        output_file = args.output
    else:
        dataset_name = os.path.basename(args.dataset_path.rstrip("/"))
        output_file = f"{dataset_name}_{args.split}.jsonl"

    print(f"Saving to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print("Done ✅")


if __name__ == "__main__":
    main()
