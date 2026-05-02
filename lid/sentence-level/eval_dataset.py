from datasets import Dataset
import sys
import json

SOURCES_COUNT = len(sys.argv) - 1
DATASET_NAME = sys.argv[-1]
LOMBARD_DIALECTS = [
    'BERGDUC', 'BREMOD', 'CRES', 'LOCC', \
    'LORUNIF', 'LSI', 'MILCLASS', 'NOL', 'SL'
]

entries_all = []
for i in range(1, SOURCES_COUNT):
    source_path = sys.argv[i]
    with open(source_path, 'r', encoding='utf-8') as f:
        source_entries = [ json.loads(line) for line in f ]
    entries_all.extend([ (entry['tag'], entry['text']) for entry in source_entries ])

dataset = {
    'tag': [ tag for (tag, _) in entries_all ],
    'text': [ text for (_, text) in entries_all ]
}

def rename_lombard(labels):
    renamed = []
    for label in labels:
        if label in LOMBARD_DIALECTS:
            label = 'LMO'
        renamed.append(label)
    return renamed

dataset['tag'] = rename_lombard(dataset['tag'])
dataset = Dataset.from_dict(dataset)
dataset.save_to_disk(DATASET_NAME)
