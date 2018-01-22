import numpy
import cPickle
import theano
import theano.tensor as tensor
import theano.tensor.shared_randomstreams as RS
from theano import config as cf
import load_data

dataset = 'data/authors.pkl'
train, valid, test = load_data.load(dataset)

def ngram2(seqs, vocab_size=1000):
    freq_pair = numpy.zeros((vocab_size, vocab_size))
    freq_w = numpy.zeros((vocab_size,))

    for seq in seqs:
        for i, w in enumerate(seq):
            if w >= vocab_size: w1 = 0
            else: w1 = w

            if i == len(seq) - 1:
                freq_w[w1] += 1
                continue

            if seq[i + 1] >= vocab_size: w2 = 0
            else: w2 = seq[i + 1]

            freq_pair[w1, w2] += 1
            freq_w[w1] += 1

    freq_pair = freq_pair * 1.0 / freq_w[:, None]
    return freq_pair, freq_w