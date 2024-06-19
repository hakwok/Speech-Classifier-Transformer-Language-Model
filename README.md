# Speech-Transformer-NLP-Classifier

## Description

This repository explores transformer architecture using speeches from US politicians. The project involves creating and training transformer encoder and decoder models. The first task predicts the speaker of a speech segment using a transformer encoder and classifier. The second task pretrains a word-level transformer decoder for language modeling, using masked self-attention and perplexity measures. The dataset includes speeches from George W. Bush, George H. Bush, and Barack Obama. The study covers model design, training procedures, and empirical findings, providing insights into transformer models.

## Installation

The main dependencies have been provided as import statements in the respective scripts.

Please be sure to have installed Torch, and optionally, Matplotlib. nltk will be installed upon running *main.py*

## Usage

To run the training and testing, simply enter python main.py in the terminal. It will go through each part sequentially using the default hyperparameters.

To run a specific part of the PA, please add --part *x*, with x being the specific part (1, 2, 3) to run.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. To view a copy of this license, visit [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
