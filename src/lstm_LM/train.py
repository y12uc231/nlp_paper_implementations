#!/usr/bin/env python
# coding: utf-8
# Author : Satyapriya Krishna
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
import numpy as np
import random
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm
from torchsummary import summary
import pickle
from typing import Tuple
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel


# Set the seed for PyTorch
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.cuda.manual_seed_all(7)  
np.random.seed(7)
random.seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
D_MODEL = 512
NUM_HIDDEN_LAYERS = 1
DIM_FEEDFORWARD = 512
LEARNING_RATE = 0.01
BATCH_SIZE = 5120
EPOCHS = 35
SEQUENCE_LENGTH = 60
MODEL_FILE_NAME = "language_model.pth"
SAVED_TOKENIZER_FILE = "tokenizer.json"
TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
SLICED_TRAIN_DATA_LOC = "train_{}.pt".format(SEQUENCE_LENGTH)
SLICED_TEST_DATA_LOC = "test_{}.pt".format(SEQUENCE_LENGTH)



#Tokenizer Builder
def build_and_save_tokenizer(file_path, tokenizer_path):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(special_tokens=["<unk>", "<sos>", "<eos>"])
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    tokenizer.train_from_iterator(lines, trainer)
    tokenizer.save(tokenizer_path)
    return tokenizer



# Define tokenizer 
def get_tokenizer(tokenizer_fname=SAVED_TOKENIZER_FILE):
    if os.path.exists(tokenizer_fname):
        print("Tokenizer found!") 
        tokenizer = Tokenizer.from_file(tokenizer_fname)
        tokenizer.end_token = "<eos>"
        tokenizer.end_token_id = tokenizer.token_to_id(tokenizer.end_token)
        tokenizer.generation_start_token = "<sos>"
        tokenizer.generation_start_token_id = tokenizer.token_to_id(tokenizer.generation_start_token)
    else:
        tokenizer = build_and_save_tokenizer(TRAIN_FILE, SAVED_TOKENIZER_FILE)
        tokenizer.end_token = "<eos>"
        tokenizer.end_token_id = tokenizer.token_to_id(tokenizer.end_token)
        tokenizer.generation_start_token = "<sos>"
        tokenizer.generation_start_token_id = tokenizer.token_to_id(tokenizer.generation_start_token)
    return tokenizer

def encode_text(tokenizer, text, max_length = SEQUENCE_LENGTH):
    encoded = tokenizer.encode(text)
    length_seq = len(encoded.ids[:])
    encoded_ids = encoded.ids[:]
    remaining = [tokenizer.end_token_id]
    repeats_needed = SEQUENCE_LENGTH - length_seq
    repeated_slice = remaining*repeats_needed
    encoded_ids.extend(repeated_slice)
    # print(f"Encoded sentence : {encoded_ids} \t", len(encoded_ids))
    return torch.tensor(encoded_ids), length_seq


tokenizer = get_tokenizer()
VOCAB_SIZE = tokenizer.get_vocab_size()
print("Vocab Size : ", VOCAB_SIZE)
END_TOKEN_ID = tokenizer.end_token_id
GENERATION_START_TOKEN_ID = tokenizer.generation_start_token_id


# Define Dataloader

class LMDataset(Dataset):
    def __init__(self, file_path, store_file_path,max_length = SEQUENCE_LENGTH):
        self.lines = []
        num_lines = 100000
        max_length_dataset = 0
        if os.path.exists(store_file_path):
            self.lines = torch.load(store_file_path)
        else:                
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in tqdm(file, desc = "Dataset Prep"):
                    line = "<sos>" + line.strip() + "<eos> "
                    line,length_seq = encode_text(tokenizer, line)
                    if length_seq > max_length_dataset:
                        max_length_dataset = length_seq
                    self.lines.append(line)
                    # num_lines -= 1

            print("Max length in the dataset : ", max_length_dataset)
            torch.save(self.lines, store_file_path)       
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        return line


    
def collate_train_fn(batch):
    batch = torch.stack(batch)
    input_seq = batch[:, :-1]
    target_seq = batch[:, 1:]
    return input_seq, target_seq



train_dataset = LMDataset(TRAIN_FILE, SLICED_TRAIN_DATA_LOC)
test_dataset = LMDataset(TEST_FILE, SLICED_TEST_DATA_LOC)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = collate_train_fn)
test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn = collate_train_fn) 



# Model Definition 
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state
        
    def init_state(self, batch_size):
        return (torch.zeros(self.lstm.num_layers,batch_size, self.lstm.hidden_size),
                torch.zeros(self.lstm.num_layers,batch_size, self.lstm.hidden_size))
    
    
# Initialize the model
model = RNNLanguageModel(VOCAB_SIZE, D_MODEL, DIM_FEEDFORWARD, NUM_HIDDEN_LAYERS)
# Load the state dictionary
if os.path.exists(MODEL_FILE_NAME):
    state_dict = torch.load(MODEL_FILE_NAME)
    model.load_state_dict(state_dict)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Evaluation
def generate_text(model, start_seq, max_length, tokenizer, target, device):
    model.eval()
    words_ids = tokenizer.encode(start_seq).ids[:]
    state_h, state_c = model.init_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    
    for i in range(max_length):
        x = torch.tensor(words_ids).unsqueeze(0).to(device) 
        logits, (state_h, state_c) = model(x, (state_h, state_c))
        prediction = torch.argmax(logits[:, -1, :], dim=1).item()
        words_ids.append(prediction)
        if prediction == tokenizer.end_token_id:
            break
    generation = tokenizer.decode(words_ids).replace(' ', '').split("=")[1].strip()
    #print(f"Evaluation:{start_seq}{generation}|| TARGET:{target}|| Answer:{int(generation.strip() == target.strip())}")
    return int(generation.strip() == target.strip())

def compute_accuracy(file_name):
    test_lines_left = []
    test_lines_right = []
    max_length = 60
    num_samples = 0
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            left_side = line.strip().split("=")[0]
            test_lines_left.append("<sos>" + left_side +"=")
            right_side = line.strip().split("=")[1]
            test_lines_right.append(right_side)
            num_samples += 1
            if num_samples > 10500:
                break
    accuracy = 0
    for left_side, right_side in tqdm(zip(test_lines_left, test_lines_right), desc="Computing Accuracy.."):
        accuracy += generate_text(model, left_side, max_length, tokenizer, right_side, device)
        
#     return accuracy/len(test_lines_left)
    return accuracy/num_samples

# Define the training loop
def train(model, train_loader, criterion, optimizer, num_epochs, test_loader, device):
    model.train()
    accuracy_epochs = []
    for epoch in range(num_epochs):
        state_h, state_c = model.init_state(BATCH_SIZE)
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        
        for batch, (x, y) in enumerate(tqdm(train_loader)):
#             print(x.shape)
            if x.size(0) < BATCH_SIZE:
                state_h, state_c = model.init_state(x.size(0))
                state_h = state_h.to(device)
                state_c = state_c.to(device)
                
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            
            logits, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y)
            
            state_h = state_h.detach()
            state_c = state_c.detach()
            
            loss.backward()
            optimizer.step()
            
            if batch % 200 == 0:
                #print(evaluate(model, test_loader, device))
                #model.train()
                torch.save(model.state_dict(), MODEL_FILE_NAME)
                print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})
        accuracy_test = compute_accuracy(TEST_FILE)
        accuracy_epochs.append(accuracy_test)
        print("Test Accuracy : ", accuracy_test)
        model.train()
    return  accuracy_epochs   


# Define the evaluation loop
def evaluate(model, test_loader, device):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            y = y.to(device)
            state_h, state_c = model.init_state(x.size(0))
            state_h = state_h.to(device)
            state_c = state_c.to(device)
        
            logits, _ = model(x, (state_h, state_c))
            predicted = torch.argmax(logits, dim=2)
            total_acc += (predicted == y).sum().item()
            total_count += y.numel()
    return total_acc / total_count





# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
accuracy_epochs = train(model, train_dataloader, criterion, optimizer, EPOCHS, test_dataloader, device)

print("Test Accuracy: ", accuracy_epochs)
torch.save(model.state_dict(), MODEL_FILE_NAME)


