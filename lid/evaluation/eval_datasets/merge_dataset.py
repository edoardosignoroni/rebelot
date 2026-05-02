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

labels_tokens = []
for dataset_path in dataset_paths:
    print(dataset_path)
    dataset = load_from_disk(dataset_path)
    print(len(dataset))
    for entry in dataset:
        labels_tokens.append((entry['labels'], entry['tokens']))

random.shuffle(labels_tokens)
labels = [ labels for (labels, tokens) in labels_tokens ]
tokens = [ tokens for (labels, tokens) in labels_tokens ]

print(len(labels))
print(len(tokens))

dataset = Dataset.from_dict({
    'labels': labels,
    'tokens': tokens
})
print(dataset)
dataset.save_to_disk(DATASET_NAME)
