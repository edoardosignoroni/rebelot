import nltk
import re
from nltk.tokenize import word_tokenize
import sys
from datasets import Dataset
import json

jsonl_file = sys.argv[1]
dataset_name = sys.argv[2]

langs_labels = {
    "<lmo>": "LMO",
    "<eng>": "ENG",
}

starts_ends = {
    "<lmo>": "</lmo>",
    "<eng>": "</eng>",
}

def get_spans(part):
    results = []
    i = 0
    curr_span = ""
    curr_label = None
    while i < len(part):
        start = None
        for start_tag in langs_labels.keys():
            if part[i:].startswith(start_tag):
                start = start_tag
                break
        if start is not None:
            if curr_span != "":
                results.append((curr_span, curr_label))
            curr_span = ""
            curr_label = langs_labels[start]
            i += 5
            while not part[i:].startswith(starts_ends[start]):
                curr_span += part[i]
                i += 1
            i += 7
            results.append((curr_span, curr_label))
            curr_span = ""
            continue
        curr_span += part[i]
        curr_label = "ITA"
        i += 1
    results.append((curr_span, curr_label))
    return results

def weighted_split(lst, weights):
    if not abs(sum(weights) - 1.0) < 1e-9:
        raise ValueError("Weights must sum to 1.0")

    n = len(lst)
    splits = []
    start = 0

    for w in weights[:-1]:
        size = round(w * n)
        splits.append(lst[start:start + size])
        start += size

    # last chunk gets the rest
    splits.append(lst[start:])
    return splits

with open(jsonl_file, 'r', encoding='utf-8') as f:
    entries = []
    for line in f:
        data = json.loads(line)
        text = data['text']
        tokens = []
        labels = []
        spans = get_spans(text)
        for text, label in spans:
            tokenized_span = word_tokenize(text)
            for token in tokenized_span:
                curr_label = label if token.isalpha() else "XXX"
                tokens.append(token)
                labels.append(curr_label)
        entry = {'tokens': tokens, 'labels': labels}
        entries.append(entry)

dataset = Dataset.from_list(entries)
dataset.save_to_disk(dataset_name)
