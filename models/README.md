# Baseline Model Fine-Tuning and Evaluation

To establish baseline performance metrics of models using this synthetic code, there are four models defined in this repo:
* Token-level classification using BETO (Spanish BERT) for sequence classification with only the existing COWS-L2H corpus as training data
* Token-level classification using BETO with COWS-L2H and my synthetically-generated corpus as training data
* Seq2seq classification (as in NMT-inspired models) using BARTO (Spanish BART) with the existing COWS-L2H corpus as training data
* Seq2seq classification (as in NMT-inspired models) using BARTO with COWS-L2H and my synthetically-generated corpus as training data

Therefore, this is a 2x2 experimental design:

| Model | COWS-L2H | COWS-L2H + Synthetic Dataset |
| ---- | ---- | ----------- |
| BETO (Token classification) | X | X|
| BARTO (Seq2seq) | X | X |

## Running the models

The seq2seq models can be fine-tuned using the seq2seq_model.ipynb Jupyter notebook.

The token-level classification models can be fine-tuned using the token_mode.ipynb Jupyter notebook.

However, I have already fine-tuned the models for GEC, so fine-tuning yourself is only for replication. The fine-tuned models are available on HuggingFace at:

* BETO (COWS-L2H only): TBA
* BETO (COWS-L2H + Synthetic): TBA
* BARTO (COWS-L2H only): TBA
* BARTO (COWS-L2H + Synthetic): TBA

## Results

TBA