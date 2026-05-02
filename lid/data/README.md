# Data preprocessing

This folder contains scripts and data used to create a training dataset for the token-level language identification tool.

## Raw data

The Italian and English monolingual sentences were obtained from the OPUS OpenSubtitles corpus.

- ITA: https://object.pouta.csc.fi/OPUS-OpenSubtitles/v1/mono/it.txt.gz
- ENG: https://object.pouta.csc.fi/OPUS-OpenSubtitles/v1/mono/en.txt.gz

Save these files into the `raw_data` folder as `ITA_SUBTITLES.txt` and `ENG_SUBTITLES.txt`.

## Data filtering

To preprocess the collected sentences run `make filter` in this folder. This filters sentences shorter than 30 characters. It further removes undesired sentence begginings such as "- " and <i> that are frequently present in the data files.

## Lombard data

The folder `code-mixing/jsonl/` contains splits of Lombard code-mixing dataset. To convert them into a HuggingFace format dataset run `make hf-cs-dataset`. 

## Data preprocessing

Run `make splits` to preprocess the data for further processing. This creates files in the `monolingual/LANG/` folders. Each language included in the LID tool will have a separate folder, each containing jsonl files with the split entries.

