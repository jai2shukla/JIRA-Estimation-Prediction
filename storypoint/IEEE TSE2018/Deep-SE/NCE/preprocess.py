import numpy
import cPickle as pkl
import pandas
from collections import OrderedDict
import os
import re
import sys
from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold

import load_raw_text

def build_dict(sentences):
    print 'Building dictionary..'
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w == '<unk>': continue
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]

    print 'dictionary:', len(keys), counts[sorted_idx[0]], counts[sorted_idx[1]]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+1  # leave 0 for UNK

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict

def grab_data(sentences, dictionary):
    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 0 for w in words]

    return seqs

def main():
    train, valid, test = load_raw_text.load(sys.argv[1])
    dictionary = build_dict(train)

    print "after building dict..."

    train_x = grab_data(train, dictionary)
    valid_x = grab_data(valid, dictionary)
    test_x = grab_data(test, dictionary)

    f = open('data/' + sys.argv[1] + '.pkl', 'wb')
    pkl.dump((train_x, valid_x, test_x), f, -1)
    f.close()

    f = open('data/' + sys.argv[1] + '.dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()


if __name__ == '__main__':
    main()
