#!/nlp/projekty/langtok/lombard/mmbert_venv/bin/python
from datasets import load_from_disk as load_dataset
from evaluate import load
from transformers import \
    AutoTokenizer, \
    TrainingArguments, \
    Trainer, \
    DataCollatorForTokenClassification, \
    AutoConfig, \
    AutoModel, \
    AutoModelForTokenClassification
from functools import partial
import numpy as np
import torch
import sys

dataset_path = sys.argv[1]
model_checkpoint = sys.argv[2]
xxx_label = sys.argv[3].lower() == 'true'
model_name = sys.argv[4]

label_list = ['LMO', 'ITA', 'ENG']
if xxx_label:
    label_list.append('XXX')

label_to_int = { label: i for i, label in enumerate(label_list)}
int_to_label = { i: label for i, label in enumerate(label_list)}

batch_size = 16
metric = load("seqeval")

dataset = load_dataset(dataset_path)
train_ds = dataset['train']
validation_ds = dataset['validate']

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=int_to_label,
    label2id=label_to_int,
    trust_remote_code=True
)

def rename_labels(examples):
    new_labels = []
    for i, labels in enumerate(examples['labels']):
        labels_entry = []
        for j, label in enumerate(labels):
            if xxx_label and not any(c.isalpha() for c in examples['tokens'][i][j]):
                labels_entry.append('XXX')
            elif label == 'ITA':
                labels_entry.append('ITA')
            elif label == 'ENG':
                labels_entry.append('ENG')
            elif label == 'DEU':
                labels_entry.append('DEU')
            elif label == 'SWE':
                labels_entry.append('SWE')
            elif label == 'SPA':
                labels_entry.append('SPA')
            elif label == 'FRA':
                labels_entry.append('FRA')
            else:
                labels_entry.append('LMO')
        new_labels.append(labels_entry)
    examples['labels'] = new_labels
    return examples

# this method converts language ids into integer values based on the label_to_int dictionary
def convert_labels_to_int(examples):
    labels_all = examples['labels']
    new_labels = []
    for i, labels in enumerate(labels_all):
        new_labels.append([])
        for label in labels:
            new_labels[i].append(label_to_int[label])
    examples['labels'] = new_labels
    return examples

# method from Hugging Face
def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_ds = train_ds.map(rename_labels, batched=True)
validation_ds = validation_ds.map(rename_labels, batched=True)

# mapping the method convert_labels_to_int on the whole dataset
train_ds = train_ds.map(convert_labels_to_int, batched=True)
validation_ds = validation_ds.map(convert_labels_to_int, batched=True)

# mapping the method tokenize_and_align_labels on the whole dataset
tokenized_train_ds = train_ds.map(partial(tokenize_and_align_labels, tokenizer=tokenizer), batched=True)
tokenized_validation_ds = validation_ds.map(
    partial(tokenize_and_align_labels, tokenizer=tokenizer),
    batched=True
)

# from Hugging Face
args = TrainingArguments(
    model_name,
    eval_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    filtered_predictions = []
    filtered_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        for p_i, l_i in zip(pred_seq, label_seq):
            if l_i != -100:
                filtered_predictions.append(p_i)
                filtered_labels.append(l_i)

    filtered_predictions = np.array(filtered_predictions)
    filtered_labels = np.array(filtered_labels)

    cm = confusion_matrix(
        filtered_labels,
        filtered_predictions,
        labels=list(range(len(label_list)))
    )
    cm_list = cm.tolist()
    cm_data = {
        'labels': label_list,
        'matrix': cm_list,
    }

    accuracy = accuracy_score(filtered_labels, filtered_predictions)

    precision, recall, f1, support = precision_recall_fscore_support(
        filtered_labels,
        filtered_predictions,
        labels=list(range(len(label_list))),
        zero_division=0
    )

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        filtered_labels, filtered_predictions, average='micro', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        filtered_labels, filtered_predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        filtered_labels, filtered_predictions, average='weighted', zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm_data
    }

    # Per-class (full)
    for i, label_name in enumerate(label_list):
        metrics[f"{label_name}_precision"] = precision[i]
        metrics[f"{label_name}_recall"] = recall[i]
        metrics[f"{label_name}_f1"] = f1[i]
        metrics[f"{label_name}_support"] = support[i]

    return metrics

# Initiating the data collator, metric and Trainer
data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load("seqeval")

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_validation_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model()
    