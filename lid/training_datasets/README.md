# Training dataset

This folder contains the scripts to create the training datasets containing synthetic multilingual examples.

To create the training datasets run `make all` in this folder. Make sure you have preprocessed the raw data in the `data/` folder.

The command `make all` creates three versions of the training dataset: `code-mixing`, `synthetic`, and `synthetic-cm`. The first consists only of the code-mixing examples from the Rebelot corpus, the second only synthetically generated entries, and the last contains entries obtained using both of the strategies.

The resulting datasets are saved in the folder `ht_datasets` in the HuggingFace dataset format.
