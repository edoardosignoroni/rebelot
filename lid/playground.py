from datasets import load_from_disk
import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

ds = load_from_disk(sys.argv[1])
model_path = sys.argv[2]

for split in ['train', 'test', 'validate']:
    print(split)
    ds_split = ds[split]
    for i in range(len(ds_split)):
        entry = ds_split[i]
        if 'suggest' in entry["tokens"]:
            print(i)
            print(entry['tokens'])


"""
entry = ds['train'][34]

label_list = ['LMO', 'ITA', 'ENG', 'XXX']

label2id = { label: i for i, label in enumerate(label_list) }
id2label = { i: label for i, label in enumerate(label_list) }
label_list = list(label2id.keys())

model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label_list))
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenized = tokenizer(entry['tokens'], is_split_into_words=True, return_tensors="pt")
tokenizer_tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])

with torch.no_grad():   
    logits = model(**tokenized).logits
logits = logits[0]
predictions = torch.argmax(logits, dim=1).tolist()
for i in range(len(predictions)):
    print(predictions[i], tokenizer_tokens[i])
"""