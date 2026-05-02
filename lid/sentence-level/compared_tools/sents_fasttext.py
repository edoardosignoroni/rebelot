import fasttext
import sys
from datasets import load_from_disk as load_dataset
import csv

model_path = sys.argv[1]
dataset_path = sys.argv[2]
model = fasttext.load_model(model_path)
dataset = load_dataset(dataset_path)

writer = csv.writer(sys.stdout, delimiter=";")

language_code_dict = {"lmo": "LMO", "it": "ITA", 'en': 'ENG'}

def sentence_fasttext(sentence):
    prediction = model.predict(sentence)
    language_code = prediction[0][0].replace("__label__", "")
    return language_code

for entry in dataset:
    lang = sentence_fasttext(entry['text'])
    lang = language_code_dict.get(lang, "UNK")
    writer.writerow([entry['tag'], lang, entry['text']])
