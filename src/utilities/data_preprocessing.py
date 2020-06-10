import os
import sys
import torch
import random
import torchtext
from collections import Counter
from collections import defaultdict

class Vocab():
    def __init__(self, input_text):
        self.stoi = defaultdict(int)
        self.stoi["UNK"] = 0
        self.itos = defaultdict(str)
        self.itos[0] = "UNK"
        self.freq = Counter(input_text.split())
        self.vocab_size = 1

        index_token = 1
        for string, _ in self.freq.items():
            self.stoi[string] = index_token
            self.itos[index_token] = string
            index_token += 1
            self.vocab_size += 1


def create_vocab(input):
    vocab = Vocab(input)
    return vocab


def create_vocab_torchtext(input):
    tokens = input.split()
    freq_counter = Counter(tokens)
    vocab = torchtext.vocab.Vocab(counter=freq_counter, max_size=len(freq_counter))
    return vocab



def data_split(input_lines, train_p, valid_p):
    random.shuffle(input_lines)
    input_lines = ["".join([j for j in i.lower() if (j.isalpha() == True) or (j == ' ')]) for i in input_lines]
    assert (train_p + valid_p) < 1, "Invalid train, valid proportions"
    train_text = " ".join([i[:-1] for i in input_lines[:int(train_p * len(input_lines))]])
    valid_text = " ".join([i[:-1] for i in
                           input_lines[int(train_p * len(input_lines)): int((train_p + valid_p) * len(input_lines))]])
    test_text = " ".join([i[:-1] for i in input_lines[int((train_p + valid_p) * len(input_lines)):]])
    return train_text, valid_text, test_text


def create_data_loader(input, context_length, vocab, batch_size = 32, shuffle = False):
    samples = input.split()
    i = 0
    context = []
    while i < len(samples) - context_length:
        sample = samples[i:i + context_length]
        sample = [vocab.stoi[j] for j in sample ]
        i += context_length
        for num_token, _ in enumerate(sample):
            context.append([sample[j] for j, _ in enumerate(sample) if j != num_token] + [sample[num_token]])
    return torch.utils.data.DataLoader(torch.Tensor(context).type(torch.long) , batch_size=batch_size, shuffle=shuffle)


def data_prep_from_file(file_path, batch_size, context_length = 4, train_p = 0.6, valid_p = 0.2):
    with open(file_path) as f1:
        text_lines = f1.readlines()

    train_text, valid_text, test_text = data_split(text_lines, train_p, valid_p)
    train_vocab = create_vocab(train_text)
    train_dl = create_data_loader(train_text, context_length, train_vocab, batch_size )
    valid_dl = create_data_loader(valid_text, context_length, train_vocab, batch_size)
    test_dl = create_data_loader(test_text, context_length, train_vocab, batch_size)
    return train_dl, valid_dl, test_dl, train_vocab










