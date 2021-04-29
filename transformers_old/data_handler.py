# python data_loader.py


import io
import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter

bptt = 35
batch_size = 20
val_batch_size = 10
tokenizer = get_tokenizer("basic_english")


def preprocess(iterator, vocab):

    data = [ torch.tensor( [ vocab[word] for word in tokenizer(line) ],
            dtype = torch.long) for line in iterator ]

    return torch.cat( tuple(filter( lambda t: t.numel()>0, data ) ) )



def to_batches(data, batch_size):
    n_batch = data.size(0) // batch_size
    data = data.narrow( 0, 0, n_batch * batch_size)
    data = data.view(batch_size, -1).contigous()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return data.to(device)




def get_data():

    train_iter = WikiText2(split="train")
    counter = Counter()
    for line in train_iter:
        counter.update(tokenizer(line))
    vocab = Vocab(counter)

    train_iter, val_iter, test_iter = WikiText2()

    train_data = preprocess(train_iter, vocab)
    val_data = preprocess(val_iter, vocab)
    test_data = preprocess(test_iter, vocab)

    return train_data, val_data, test_data, vocab


train_data, val_data, test_data, vocab = get_data()

print(vocab["blue"])






