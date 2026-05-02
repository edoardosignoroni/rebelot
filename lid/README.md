# Token-level language identification for Lombard

This directory contains data for training and evaluating a token-level language identification tool that recognizes three languages: Lombard, English, and Italian.

## Training

Before training the models and conducting the evaluation, make sure you have preprocessed the raw monolingual sentences and generated the training datasets. The tools and instructions for that are provided in the folders `data/` and `training_datasets/`.

To train the models run `make train` inside this directory.

# Evaluation 

To conduct the evaluation run `make eval` inside this directory. Make sure that you have created the evaluation datasets before that. The instructions and tools for that are provided in the folder `evaluation/eval_datasets`.

The results from the evaluation can be found in the `results/test-all` folder.
