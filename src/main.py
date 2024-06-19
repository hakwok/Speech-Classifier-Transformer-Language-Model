import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from torch.optim import optimizer
from transformer import *
import nltk
nltk.download("punkt")
import matplotlib.pyplot as plt
import argparse

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(encoder, classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    encoder.eval()
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            embeddings, _ = encoder(X)
            outputs = classifier(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        encoder.train()
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    criterion = nn.CrossEntropyLoss()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        Y = Y.view(-1)
        outputs, _ = decoderLMmodel(X)
        outputs = outputs.view(-1, outputs.size(-1))
        loss = criterion(outputs, Y)
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item() 

    decoderLMmodel.train()
    return perplexity

def part1(tokenizer):
    # data loading for the classification task
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    encoder = Encoder(tokenizer.vocab_size, n_embd, n_layer, n_head, 4*n_embd, block_size).to(device)
    classifier = Classifier(n_input, n_hidden, n_output).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # for the classification task, you will train for a fixed number of epochs like this:
    print("Encoder/Classifier Training")
    for epoch in range(epochs_CLS):
        total_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            # CLS training code here
            embeddings, _ = encoder(xb)
            outputs = classifier(embeddings)
            loss = criterion(outputs, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = total_loss / len(train_CLS_loader)
        train_accuracy = compute_classifier_accuracy(encoder, classifier, train_CLS_loader)
        test_accuracy = compute_classifier_accuracy(encoder, classifier, test_CLS_loader)
        print(f"Epoch #{epoch + 1}: train loss {loss:.3f} \t train accuracy {train_accuracy:.3f} \t test accuracy {test_accuracy:.3f}")

    # Sanity Check
    print("Sanity Check: Encoder Classifier")
    util = Utilities(tokenizer, encoder)
    check_sentence = "Sample sentence used for the sanity check."
    util.sanity_check(check_sentence, block_size)

def part2(tokenizer):
    # data loading for the language modeling task
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    inputfileTest = "speechesdataset/test_LM_obama.txt" # Change this to test on different files
    with open(inputfileTest, 'r', encoding='utf-8') as f:
        lmtestText = f.read()
    test_LM_dataset = LanguageModelingDataset(tokenizer, lmtestText,  block_size)
    test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=True)

    inputfileTest2 = "speechesdataset/test_LM_hbush.txt" # Change this to test on different files
    with open(inputfileTest2, 'r', encoding='utf-8') as f:
        lmtestText2 = f.read()
    test_LM_dataset2 = LanguageModelingDataset(tokenizer, lmtestText2,  block_size)
    test_LM_loader2 = DataLoader(test_LM_dataset2, batch_size=batch_size, shuffle=True)

    inputfileTest3 = "speechesdataset/test_LM_wbush.txt" # Change this to test on different files
    with open(inputfileTest3, 'r', encoding='utf-8') as f:
        lmtestText3 = f.read()
    test_LM_dataset3 = LanguageModelingDataset(tokenizer, lmtestText3,  block_size)
    test_LM_loader3 = DataLoader(test_LM_dataset3, batch_size=batch_size, shuffle=True)

    # training the decoder for the language modeling task
    decoder = Decoder(tokenizer.vocab_size, n_embd, n_layer, n_head, d_ff=100, max_seq_len=block_size).to(device)
    optimizer = optim.Adam(list(decoder.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    print("Decoder Training")
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        # LM training code here
        outputs, _ = decoder(xb)
        yb = yb.view(-1)
        outputs = outputs.view(-1, outputs.size(-1))
        loss = criterion(outputs, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            loss = loss.item()
            train_perplexity = compute_perplexity(decoder, train_LM_loader)
            print(f"Iteration #{i + 1}: train loss {loss:.3f} \t train perplexity {train_perplexity:.3f}")
    obama_perplexity = compute_perplexity(decoder, test_LM_loader)
    hbush_perplexity = compute_perplexity(decoder, test_LM_loader2)
    wbush_perplexity = compute_perplexity(decoder, test_LM_loader3)
    print("Iteration #500 Obama perplexity:", obama_perplexity)
    print("Iteration #500 H.Bush perplexity:", hbush_perplexity)
    print("Iteration #500 W.Bush perplexity:", wbush_perplexity)

    # Sanity Check 
    print("Sanity Check: LM")
    util = Utilities(tokenizer, decoder)
    check_sentence = "Sample sentence used for the sanity check."
    util.sanity_check(check_sentence, block_size)

def part3(tokenizer):
    # data loading for the classification task
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    encoder = Encoder(tokenizer.vocab_size, n_embd, n_layer, n_head, 4*n_embd, block_size, dropout=0.1).to(device)
    classifier = Classifier(n_input, n_hidden + 1000, n_output).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # for the classification task, you will train for a fixed number of epochs like this:
    print("Encoder/Classifier Training")
    for epoch in range(epochs_CLS + 15):
        total_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            # CLS training code here
            embeddings, _ = encoder(xb)
            outputs = classifier(embeddings)
            loss = criterion(outputs, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = total_loss / len(train_CLS_loader)
        train_accuracy = compute_classifier_accuracy(encoder, classifier, train_CLS_loader)
        test_accuracy = compute_classifier_accuracy(encoder, classifier, test_CLS_loader)
        print(f"Epoch #{epoch + 1}: train loss {loss:.3f} \t train accuracy {train_accuracy:.3f} \t test accuracy {test_accuracy:.3f}")

    # Sanity Check
    print("Sanity Check: Encoder Classifier with Dropout")
    util = Utilities(tokenizer, encoder)
    check_sentence = "Sample sentence used for the sanity check."
    util.sanity_check(check_sentence, block_size)

def main():
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) 
    print("Vocabulary size is", tokenizer.vocab_size)

    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, required=False, choices=['1', '2', '3'])
    args = parser.parse_args()

    if args.part == '1':
        part1(tokenizer)
    elif args.part == '2':
        part2(tokenizer)
    elif args.part == '3':
        part3(tokenizer)
    else:
        part1(tokenizer)
        part2(tokenizer)
        part3(tokenizer)

if __name__ == "__main__":
    main()
