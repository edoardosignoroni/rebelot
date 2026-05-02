from datasets import load_from_disk, Dataset, DatasetDict
import sys
import random

SEED = 42
random.seed(SEED)
DATASET_NAME = sys.argv[-1]

dataset_paths = []
for i in range(1,len(sys.argv)-1):
    print(sys.argv[i])
    dataset_paths.append(sys.argv[i])

merged_dataset = {split: [] for split in ['train', 'validate', 'test']}
for dataset_path in dataset_paths:
    print(dataset_path)
    dataset = load_from_disk(dataset_path)
    for split in merged_dataset:
        print(split)
        labels_tokens = []
        for entry in dataset[split]:
            labels_tokens.append((entry['labels'], entry['tokens']))
        random.shuffle(labels_tokens)
        merged_dataset[split].extend(labels_tokens)

for key, value in merged_dataset.items():
    print(key)

merged_dataset_dict = DatasetDict()
for split in merged_dataset:
    entries = []
    for (labels, tokens) in merged_dataset[split]:
        entries.append({'labels': labels, 'tokens': tokens})
    dataset = Dataset.from_dict({
        'tokens': [ entry['tokens'] for entry in entries ],
        'labels': [ entry['labels'] for entry in entries ]
    })
    merged_dataset_dict[split] = dataset

print(merged_dataset_dict)
print(merged_dataset_dict['train'][100])
merged_dataset_dict.save_to_disk(DATASET_NAME)