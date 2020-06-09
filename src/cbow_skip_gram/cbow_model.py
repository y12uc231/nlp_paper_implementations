import torch
import torch.nn as nn
import torch.optim as optim

class CBOW(nn.Module):
    def __init__(self):
        super(CBOW, self).__init__()
        self.test = 1

    def forward(self, x):
        return x


