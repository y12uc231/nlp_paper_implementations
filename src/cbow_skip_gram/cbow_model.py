import torch
import torch.nn as nn
<<<<<<< HEAD
import torch.optim as optim

class CBOW(nn.Module):
    def __init__(self):
        super(CBOW, self).__init__()
        self.test = 1

    def forward(self, x):
        return x
=======
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dims):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dims)
        self.fc = nn.Linear(embedding_dims, vocab_size)

    def forward(self, x):
        x = F.tanh(self.embeddings(x).mean(0))
        x = F.softmax(self.fc(x))


>>>>>>> Added strcture files for cbow


