# Comparison on sentence-level language prediction

The token-level LID tools were compared against two different tools for language identification that support Lombard: [FastText](https://fasttext.cc/docs/en/language-identification.html) and [GlotLID](https://huggingface.co/cis-lmu/glotlid). To run the comparison, download the two models and move the token-level LID models into the `tools_models/` folder following this structure:

tools_models/
    fasttext/
        model.bin
    glot-lid/
        model.bin
    langtok/
        langtok_models

## Evaluation dataset

Run `make eval-dataset` to preprocess the data for the evaluation and save them in a HuggingFace dataset format.

## Running the evaluation

Run `make tools` to run the evaluation of the selected LID tools.
