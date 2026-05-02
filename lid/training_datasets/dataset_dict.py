from datasets import load_from_disk, Dataset, DatasetDict
import sys

TRAIN_DATASET_PATH = sys.argv[1]
VALID_DATASET_PATH = sys.argv[2]
TEST_DATASET_PATH = sys.argv[3]
OUTPUT_DATASET_PATH = sys.argv[4]

def merge_datasets(paths):
    datasets = {}
    for split, path in paths.items():
        datasets[split] = load_from_disk(path)
    merged_dataset = DatasetDict(datasets)
    return merged_dataset

dataset_paths = {
    'train': TRAIN_DATASET_PATH,
    'validate': VALID_DATASET_PATH,
    'test': TEST_DATASET_PATH,
}

merged_dataset = merge_datasets(dataset_paths)
merged_dataset = merged_dataset.save_to_disk(OUTPUT_DATASET_PATH)
