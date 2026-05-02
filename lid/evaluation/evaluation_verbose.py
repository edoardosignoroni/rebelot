# -*- coding: utf-8 -*-

import json
import sys
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict, load_from_disk as load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import torch
import csv
import numpy as np

model_path = sys.argv[1]
dataset_path = sys.argv[2]
xxx_label = model_path.lower().endswith('-true')
mistakes_file = sys.argv[3]

label_list = ['LMO', 'ITA', 'ENG']
if xxx_label:
    label_list.append('XXX')

label2id = { label: i for i, label in enumerate(label_list) }
id2label = { i: label for i, label in enumerate(label_list) }
label_list = list(label2id.keys())

model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label_list))
tokenizer = AutoTokenizer.from_pretrained(model_path)
seqeval = evaluate.load("seqeval")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True)
    labels = []
    for i, label in enumerate(examples[f"labels"]):
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

def convert_labels_to_int(examples):
    dataset_labels = label2id.copy()
    if not xxx_label:
        dataset_labels['XXX'] = -100
    labels_all = examples['labels']
    new_labels = []
    for i, labels in enumerate(labels_all):
        new_labels.append([])
        for label in labels:
            new_labels[i].append(dataset_labels[label])
    examples['labels'] = new_labels
    return examples

def process_sentence(sentence):
    tokenized = tokenizer(sentence, is_split_into_words=True, return_tensors="pt")
    tokenizer_tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])
    if len(tokenized['input_ids'][0]) > tokenizer.model_max_length:
        return None
    with torch.no_grad():   
        logits = model(**tokenized).logits
    logits = logits[0]
    predictions = torch.argmax(logits, dim=1).tolist()
    return predictions, tokenizer_tokens

def eval_on_dataset(tokenized_ds):
    with open(mistakes_file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        # confusion matrix
        selected_langs = [ [ 0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys())) ]
        mistakes = []
        sentences_count = tokenized_ds.num_rows
        tokens_all = 0
        tokens_evaluated = 0
        skipped = 0
        for i in range(sentences_count):
            # printing the number of the sentence that is currently being processed to give a better picture about the process of testing
            tokens = tokenized_ds[i]['tokens']
            s = process_sentence(tokens)
            if s is None:
                continue
            predicted_labels, tokenizer_tokens = s
            tokens_all += len(tokenizer_tokens)
            correct_labels = tokenized_ds[i]['labels']
            writer.writerow([' '.join(tokens), '', '', ''])
            assert predicted_labels is not None, "Predicted labels are None, check the sentence length."
            for j in range(len(predicted_labels)):
                correct = correct_labels[j]
                predicted = predicted_labels[j]
                correct_label = id2label[correct] if correct != -100 else 'XXX'
                predicted_label = id2label[predicted]
                mistake_mark = 'O' if correct == predicted else 'X'
                writer.writerow([mistake_mark, correct_label, predicted_label, tokenizer_tokens[j]])
                if correct == -100 or tokenizer_tokens[j] == "\u2581" and not 'XXX' in label_list:
                    skipped += 1
                    continue
                tokens_evaluated += 1
                if correct != predicted:
                    mistake_line = {
                        'sentence_id': i, 'token_id': j,
                        'tokens': tokens, 'correct_labels': correct_labels,
                        'predicted_labels': predicted_labels, 'token': tokenizer_tokens[j],
                        'correct': id2label[correct], 'predicted': id2label[predicted]
                    }
                    mistakes.append(mistake_line)
                selected_langs[correct][predicted] += 1
        return selected_langs, mistakes

def classification_metrics(confusion_matrix):
    """
    Compute accuracy, precision, recall, and F1 score
    from a confusion matrix (list of lists or np.array).

    Returns
    -------
    dict containing:
        - per_class metrics
        - macro / micro / weighted averages
    """

    cm = np.asarray(confusion_matrix)
    assert cm.ndim == 2 and cm.shape[0] == cm.shape[1]

    # Overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)

    # Per-class statistics
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP

    precision_per_class = np.divide(
        TP, TP + FP,
        out=np.zeros_like(TP, dtype=float),
        where=(TP + FP) != 0
    )

    recall_per_class = np.divide(
        TP, TP + FN,
        out=np.zeros_like(TP, dtype=float),
        where=(TP + FN) != 0
    )

    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(precision_per_class, dtype=float),
        where=(precision_per_class + recall_per_class) != 0
    )

    support = cm.sum(axis=1)

    result = {}

    # Store per-class metrics
    result["per_class"] = {
        "precision": precision_per_class.tolist(),
        "recall": recall_per_class.tolist(),
        "f1": f1_per_class.tolist(),
        "support": support.tolist()
    }

    # Averages
    for average in ['macro', 'micro', 'weighted']:

        if average == "macro":
            precision = precision_per_class.mean()
            recall = recall_per_class.mean()
            f1 = f1_per_class.mean()

        elif average == "weighted":
            precision = np.average(precision_per_class, weights=support)
            recall = np.average(recall_per_class, weights=support)
            f1 = np.average(f1_per_class, weights=support)

        elif average == "micro":
            TP_micro = TP.sum()
            FP_micro = FP.sum()
            FN_micro = FN.sum()

            precision = TP_micro / (TP_micro + FP_micro) if (TP_micro + FP_micro) > 0 else 0.0
            recall = TP_micro / (TP_micro + FN_micro) if (TP_micro + FN_micro) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result[average] = {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return result

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


raw_eval_ds = load_dataset(dataset_path)
if isinstance(raw_eval_ds, dict):
    raw_eval_ds = raw_eval_ds['test']

raw_eval_ds = raw_eval_ds.map(rename_labels, batched=True)
raw_eval_ds = raw_eval_ds.map(convert_labels_to_int, batched=True)
tokenized_eval_ds = raw_eval_ds.map(tokenize_and_align_labels, batched=True)

selected_langs, mistakes = eval_on_dataset(tokenized_eval_ds)
metrics = classification_metrics(np.array(selected_langs))
sys.stdout.write(json.dumps({
    'metrics': metrics,
    'confusion_matrix': selected_langs
}, indent=4, ensure_ascii=False))