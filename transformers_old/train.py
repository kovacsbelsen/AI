import io
import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import time
import torch.nn as nn
import math
import numpy as np

from models import TransformerModel, PositionalEncoding
import data_handler as dh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_data, val_data, test_data, vocab = dh.get_data()

n_tokens = len(vocab.stoi)
emb_size = 512
n_hidden = 200
n_layers = 2
n_heads = 2
dropout = 0.2
lr = 5.0

model = TransformerModel(n_tokens, emb_size, n_heads, n_hidden, n_layers, dropout).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

def train(model):

    model.train()
    total_loss = 0.
    start_time = time()

    src_mask = model.generate_square_subsequent_mask(dh, bptt)

    for batch, i in enumarate(0, train_data.size(0) -1 , dh.bptt):

        data, targets = dh.get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != dh.bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0))

        output = model(data, src_mask)   
        loss = criterion(output.view(-1), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        total_loss += loss