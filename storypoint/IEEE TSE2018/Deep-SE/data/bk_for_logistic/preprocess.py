#
#
#
import cPickle as pkl
import numpy
import sys
from sklearn.cross_validation import StratifiedKFold
import gzip

import load_raw_text

from subprocess import Popen, PIPE

remove_rare_classes = True

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['C:/Perl64/perl/bin/perl', 'tokenizer.perl', '-l', 'en', '-q', '-']

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

    print 'dictionary:', len(keys), counts[sorted_idx[0]], counts[sorted_idx[1]]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+1  # leave 0 (UNK)

    print numpy.sum(counts), ' total words ', sum(counts[sorted_idx[:1000]]), ' frequency of the 1000th frequent word'

    return worddict

def grab_data(title, description, dictionary):
    title = tokenize(title)
    description = tokenize(description)

    seqs = [[None] * len(title), [None] * len(description)]
    for i, sentences in enumerate([title, description]):
        for idx, ss in enumerate(sentences):
            words = ss.strip().lower().split()
            seqs[i][idx] = [dictionary[w] if w in dictionary else 0 for w in words]

    return seqs[0], seqs[1]

def main():
    data_path = sys.argv[1] + '.csv'
    title, description, labels, rare_classes = load_raw_text.load(data_path)

    if not remove_rare_classes: rare_classes = []
    classes = list(set(labels) - set(rare_classes))
    classes.sort()

    cdict = {}
    for i, c in enumerate(classes):
        cdict[c] = i

    f = open(sys.argv[1] + '_3sets.txt', 'r')
    train_ids, valid_ids, test_ids = [], [], []
    count = -2
    n_rare = 0
    for line in f:
        if count == -2:
            count += 1
            continue

        count += 1
        if labels[count] in rare_classes:
            n_rare += 1
            continue
        ls = line.split()
        if ls[0] == '1': train_ids.append(count)
        if ls[1] == '1': valid_ids.append(count)
        if ls[2] == '1': test_ids.append(count)

    print 'nrare: ', n_rare
    print 'ntrain, nvalid, ntest: ', len(train_ids), len(valid_ids), len(test_ids)

    for i, label in enumerate(labels):
        if label in cdict:
            labels[i] = cdict[label]

    train_title, train_description, train_labels = title[train_ids], description[train_ids], labels[train_ids]
    valid_title, valid_description, valid_labels = title[valid_ids], description[valid_ids], labels[valid_ids]
    test_title, test_description, test_labels = title[test_ids], description[test_ids], labels[test_ids]

    dictionary = build_dict(numpy.concatenate([train_title, train_description]))

    print "after building dict..."
    train_t, train_d = grab_data(train_title, train_description, dictionary)
    valid_t, valid_d = grab_data(valid_title, valid_description, dictionary)
    test_t, test_d = grab_data(test_title, test_description, dictionary)

    f = gzip.open(sys.argv[1] + '.pkl.gz', 'wb')
    pkl.dump((train_t, train_d, train_labels,
              valid_t, valid_d, valid_labels,
              test_t, test_d, test_labels), f, -1)
    f.close()

    f = gzip.open(sys.argv[1] + '.dict.pkl.gz', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()


if __name__ == '__main__':
    main()
