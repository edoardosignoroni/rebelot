import sys
import json
from nltk.tokenize import word_tokenize
from datasets import Dataset 

infile = sys.argv[1]

entries = []
with open(infile, 'r', encoding='utf-8') as f:
    data = [ json.loads(line) for line in f ]

for entry in data:
    label = entry['tag']
    tokens = word_tokenize(entry['text'])
    labels_entry = []
    for token in tokens:
        if not any(c.isalpha() for c in token):
            labels_entry.append('XXX')
        elif label == 'ITA':
            labels_entry.append('ITA')
        elif label == 'ENG':
            labels_entry.append('ENG')
        else:
            labels_entry.append('LMO')
    entries.append({
        'tokens': tokens,
        'labels': labels_entry
    })

dataset = Dataset.from_list(entries)
dataset.save_to_disk(sys.argv[2])
