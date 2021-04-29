# Train the model = end of training test loss 5.26 , test ppl 194.11 , end of epoch   3 | time: 175.61s | valid loss  5.36 | valid ppl   212.33

# The output and the embeddings are 2 different things!

# Can you figure out how to get the embeddings?

# take a look at the shape of the target 
# take a look at the shape of the output
# take a look at the shape of the embedding


# for "yellow"  and "blue" extract the tokens
# put them into a tensor
# get the embeddings of those 2 words
# use np.inner to get the semantic simmilarity 
# what about blue and car?

# can you come up with a different way of computing the semantic simmilarity?

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

#load the model
model = torch.load("transformer_model1.pth")
#print(model)

#get the vocab from the data handler
_, _, _, vocab = dh.get_data()

#assigns the vocab index of the word
blue = vocab['blue']
yellow = vocab['yellow']
car = vocab['car']

#define the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#putting the index to a tensor
blue = torch.tensor(blue).to(device)
yellow = torch.tensor(yellow).to(device)
car = torch.tensor(car).to(device)

#get the embeddings of the word of the word
blue = model.encoder(blue)
yellow = model.encoder(yellow)
car = model.encoder(car)

#put the tensor into numpy array (they also need to be in the cpu)
blue = blue.detach().cpu().numpy()
yellow = yellow.detach().cpu().numpy()
car = car.detach().cpu().numpy()

#np.inner returns the cosine distance, that we use to measure the semantic similarity
print("Blue and yellow: ", np.inner(blue, yellow))
print("Blue and car: ", np.inner(blue, car))
print("Blue and blue: ", np.inner(blue, blue))


#try to check how similar a sentence is to a word

sentence = "This restaurant makes a very good soup"
category = "Food"