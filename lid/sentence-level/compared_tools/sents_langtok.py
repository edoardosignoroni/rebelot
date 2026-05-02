#!/nlp/projekty/langtok/langtok_env/bin/python3

import sys
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_from_disk as load_dataset
import torch
from collections import Counter
import csv

model_path = sys.argv[1]
dataset_path = sys.argv[2]

writer = csv.writer(sys.stdout, delimiter=";")

label_list = ['LMO', 'ITA', 'ENG']
if model_path.endswith("--True"):
    label_list.append('XXX')
id2label = { i: label for i, label in enumerate(label_list) }
label2id = { label: i for i, label in enumerate(label_list) }

model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label_list))
tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset = load_dataset(dataset_path)
result_string = ""
for entry in dataset:
    tokenized_input = tokenizer(entry['text'], is_split_into_words=False, return_tensors="pt")
    if len(tokenized_input) > tokenizer.model_max_length:
        writer.writerow([entry['tag'], 'UNK', entry['text']])
        print(f"{entry['tag']};UNK;{entry['text']}")
        continue
    with torch.no_grad():
        logits = model(**tokenized_input).logits
    logits = logits[0]
    predictions = torch.argmax(logits, dim=1).tolist()
    predicted_labels = list(map(lambda x: id2label[x], predictions))
    predicted_labels = list(filter(lambda x: x != "XXX", predicted_labels))
    counter = Counter(predicted_labels)
    lang = counter.most_common(1)[0][0]
    writer.writerow([entry['tag'], lang, entry['text']])

    
