import os
import sys
import torchtext
from collections import Counter


def create_vocab(input):
    tokens = input.split()
    freq_counter = Counter(tokens)
    vocab = torchtext.vocab.Vocab(counter= freq_counter, max_size= len(freq_counter) )
    return vocab

def create_samples_lm(input, context_length):
    samples = []

