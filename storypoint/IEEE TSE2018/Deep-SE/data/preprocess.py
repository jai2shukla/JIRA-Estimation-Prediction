#
#
#

chosen_frequency = 30
import cPickle as pkl
import numpy
import sys
from sklearn.cross_validation import StratifiedKFold
import gzip

import load_raw_text

from subprocess import Popen, PIPE

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['/usr/bin/perl', 'tokenizer.perl', '-l', 'en', '-q', '-']

def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    return toks

def build_dict(sentences):
    sentences = tokenize(sentences)

    print 'Building dictionary..'
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]
    counts = numpy.array(counts)

    print 'number of words in dictionary:', len(keys)

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+1  # leave 0 (UNK)

    pos = 0
    for i, c in enumerate(sorted_idx):
        if counts[c] >= chosen_frequency: pos = i

    print numpy.sum(counts), ' total words, ', pos, 'words with frequency >=', chosen_frequency

    return worddict

def grab_data(title, description, dictionary):
    title = tokenize(title)
    description = tokenize(description)

    seqs = [[None] * len(title), [None] * len(description)]
    for i, sentences in enumerate([title, description]):
        for idx, ss in enumerate(sentences):
            words = ss.strip().lower().split()
            seqs[i][idx] = [dictionary[w] if w in dictionary else 0 for w in words]
            if len(seqs[i][idx]) == 0:
                print 'len 0: ', i, idx

    return seqs[0], seqs[1]

def main():
    # load pretrain text:
    pretrain_path = sys.argv[1] + '_pretrain.csv'
    pre_title, pre_descr = load_raw_text.load_pretrain(pretrain_path)
    print 'number of datapoints:', len(pre_title)
    print "after building dict..."

    n_train = len(pre_title) * 2 // 3
    ids = numpy.arange(len(pre_title))
    numpy.random.shuffle(ids)
    train_ids = ids[:n_train]
    valid_ids = ids[n_train:]
    train = numpy.concatenate([pre_title[train_ids], pre_descr[train_ids]])
    valid = numpy.concatenate([pre_title[valid_ids], pre_descr[valid_ids]])
    dictionary = build_dict(train)
    pre_train, pre_valid = grab_data(train, valid, dictionary)
    f_pre = gzip.open(sys.argv[1] + '_pretrain.pkl.gz', 'wb')
    pkl.dump((pre_train, pre_valid, pre_valid), f_pre, -1)
    f_pre.close()

    f = gzip.open(sys.argv[1] + '.dict.pkl.gz', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()


if __name__ == '__main__':
    main()
