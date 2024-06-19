# Speech-Transformer-NLP-Classifier

## Description

This repository explores transformer architecture using speeches from US politicians. The project involves creating and training transformer encoder and decoder models. The first task predicts the speaker of a speech segment using a transformer encoder and classifier. The second task pretrains a word-level transformer decoder for language modeling, using masked self-attention and perplexity measures. The dataset includes speeches from George W. Bush, George H. Bush, and Barack Obama. The study covers model design, training procedures, and empirical findings, providing insights into transformer models.

## Installation

Ensure you have the following dependencies installed:
- [Torch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/) (optional)
- [nltk](https://www.nltk.org/)

To install `nltk`, it will be automatically installed upon running `main.py`.

## Usage

To run the training and testing, simply execute the following command in your terminal:
```sh
python src/main.py
```

This will go through each part sequentially using the default hyperparameters.

To run a specific part of the project, use the --part flag followed by the part number (1, 2, or 3):

```bash
python src/main.py --part x
```

Replace x with the specific part number you wish to run.

## Dataset

The dataset includes speeches from:

- George W. Bush
- George H. Bush
- Barack Obama

Files

- src/main.py: Main script to run the training and testing.
- src/dataset.py: Script to handle the dataset.
- src/tokenizer.py: Script to tokenize the speeches.
- src/transformer.py: Implementation of the transformer model.
- src/utilities.py: Utility functions used across the project.
- speechesdataset/: Directory containing the speech data files.
- speech_transformer_nlp_classifier_report.pdf: Report detailing the project findings.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. To view a copy of this license, visit [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
