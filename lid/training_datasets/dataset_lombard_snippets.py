import sys
from nltk.tokenize import word_tokenize
from datasets import Dataset, DatasetDict
import random
import json
import numpy as np

SEED = 42
random.seed(SEED)

MONOLINGUAL_PERCENTAGE = float(sys.argv[1])  # e.g., '0.0'
LANGUAGES = sys.argv[2]  # e.g., 'LMO-ITA'
PATH_TO_DATA = sys.argv[3]

# this code always inserts only one snippet per sentence
SNIPPETS_COUNT = int(sys.argv[4])
SNIPPETS_LENGTH = int(sys.argv[5])

DATASET_NAME = sys.argv[6]

SPLITS = ['train', 'valid', 'test']

LANGS = []
if 'lmo' in LANGUAGES.lower():
    LANGS.append('LMO')
if 'eng' in LANGUAGES.lower():
    LANGS.append('ENG')
if 'ita' in LANGUAGES.lower():
    LANGS.append('ITA')
if 'fra' in LANGUAGES.lower():
    LANGS.append('FRA')
if 'deu' in LANGUAGES.lower():
    LANGS.append('DEU')
if 'spa' in LANGUAGES.lower():
    LANGS.append('SPA')
if 'swe' in LANGUAGES.lower():
    LANGS.append('SWE')

def get_sentences_separated(split):
    sentences = {}
    for lang in LANGS:
        sentences[lang] = {}
        with open(f'{PATH_TO_DATA}/{lang}/{split}.jsonl', 'r', encoding='utf-8') as f:
            sentences[lang] = [json.loads(line) for line in f]
    
    sentences_separated = {}
    for lang in LANGS:
        n = len(sentences[lang])
        i1 = int(n * MONOLINGUAL_PERCENTAGE)
        sentences_separated[lang] = {}
        sentences_separated[lang]['monolingual'] = sentences[lang][:i1]
        sentences_separated[lang]['main_sents'] = sentences[lang][i1:]

    lmo_splited = np.array_split(sentences_separated['LMO']['main_sents'], len(LANGS) - 1)
    i = 0
    for lang in LANGS:
        if lang == 'LMO':
            continue    
        sentences_separated[lang]['snippets_sents'] = lmo_splited[i].tolist()
        i += 1

    for lang in LANGS:
        if lang == 'LMO':
            continue
        i = 0
        while len(sentences_separated[lang]['main_sents']) > len(sentences_separated[lang]['snippets_sents']):
            sentences_separated[lang]['snippets_sents'].append(sentences_separated[lang]['snippets_sents'][i])
            i += 1
    return sentences_separated

def get_snippet(snippet_entry):
    text = snippet_entry['text'][:-1]
    tag = snippet_entry['tag']
    tokens = word_tokenize(text)
    snippet_length = random.randint(1, max(1, min(SNIPPETS_LENGTH, len(tokens))))
    position = random.randint(0, len(tokens) - snippet_length)
    tokens = tokens[position:position + snippet_length]
    return tokens, tag

def monolignual_sentences(data):
    entries = []
    for entry in data:
        text = entry['text']
        tag = entry['tag']
        tokens = word_tokenize(text)
        entries.append({
            'tokens': tokens,
            'labels': [ tag for _ in tokens ]
        })
    return entries


def modified_sentences(main_sents, snippets_sents):
    assert len(main_sents) == len(snippets_sents)
    entries = []
    for i in range(len(main_sents)):
        sentence_tokenized = word_tokenize(main_sents[i]['text'])
        lang = main_sents[i]['tag']
        labels = [ lang for token in sentence_tokenized ]
        try:
            position = random.randint(1, len(sentence_tokenized) - 2)
        except ValueError:
            print(f"Warning: sentence too short to insert snippet: '{sentence_tokenized}'")
            position = 0
        snippet, snippet_lang = get_snippet(snippets_sents[i])
        # inserting the tokens from a snippet into a sentence
        for word in snippet:
            sentence_tokenized.insert(position, word)
            labels.insert(position, snippet_lang)
            position += 1
        entries.append({
            'tokens': sentence_tokenized,
            'labels': labels
        })
    return entries

dataset_dict = DatasetDict()
for split in SPLITS:
    sentences = get_sentences_separated(split)
    entries = []
    for lang in LANGS:
        entries.extend(monolignual_sentences(sentences[lang]['monolingual']))
        if lang == 'LMO':
            continue
        else:
            entries.extend(modified_sentences(sentences[lang]['main_sents'], sentences[lang]['snippets_sents']))

    dataset = Dataset.from_dict({
        'tokens': [ entry['tokens'] for entry in entries ],
        'labels': [ entry['labels'] for entry in entries ]
    })
    if split == 'valid':
        split = 'validate'
    dataset_dict[split] = dataset

dataset_dict.save_to_disk(DATASET_NAME)
            