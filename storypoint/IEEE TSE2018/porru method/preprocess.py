import gzip
import sys
import cPickle as pkl
import pandas
import numpy

from subprocess import Popen, PIPE
chosen_frequency = 10

try:
    project = sys.argv[1]
except:
    print 'No argument'
    project = 'mesos_porru'

def normalize(seqs):
    for i, s in enumerate(seqs):
        words = s.split()
        if len(words) < 1:
            seqs[i] = 'null'
    return seqs


print 'Project:' + project

tokenizer_cmd = ['/usr/bin/perl', 'tokenizer.perl', '-l', 'en', '-q', '-']

def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE) # pass string to perl function for tokenizing
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'
    return toks

def grab_data(context, codesnippet, dictionary):
    context = tokenize(context)
    codesnippet = tokenize(codesnippet)

    seqs = [[None] * len(context), [None] * len(codesnippet)]
    for i, sentences in enumerate([context, codesnippet]):
        for idx, ss in enumerate(sentences):
            words = ss.strip().lower().split()
            seqs[i][idx] = [dictionary[w] if w in dictionary else 0 for w in words]
            if len(seqs[i][idx]) == 0:
                print 'len 0: ', i, idx

    return seqs[0], seqs[1]

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

def clean_sen(sen):
    sen = ''.join([c if ord(c) < 128 and ord(c) > 32 else ' ' for c in sen])
    return sen

# read data from csv
data_path = project + '.csv'

data = pandas.read_csv(data_path).values
labels = data[:, 1].astype('float32')
context = normalize(data[:, 2].astype('str'))
codesnippet = normalize(data[:, 3].astype('str'))
binaryFeat = data[:, 4:].astype('float32')

for i in range(len(context)):
    if context[i] is None:
        context[i] = 'None'
    else:
        context[i] = clean_sen(context[i])

for i in range(len(codesnippet)):
    if codesnippet[i] is None:
        codesnippet[i] = 'None'
    else:
        codesnippet[i] = clean_sen(codesnippet[i])

# read 3 set of data
f = open(project + '_3sets.txt', 'r')
train_ids, valid_ids, test_ids = [], [], []
count = -2
for line in f:
    if count == -2:
        count += 1
        continue

    count += 1
    ls = line.split()
    if ls[0] == '1':
        train_ids.append(count)
    if ls[1] == '1':
        valid_ids.append(count)
    if ls[2] == '1':
        test_ids.append(count)

print 'ntrain, nvalid, ntest: ', len(train_ids), len(valid_ids), len(test_ids)

#preprocess data and packing
train_context, train_codesnippet, train_binaryFeat, train_labels = context[train_ids], codesnippet[train_ids], binaryFeat[train_ids], labels[train_ids]
valid_context, valid_codesnippet, valid_binaryFeat, valid_labels = context[valid_ids], codesnippet[valid_ids], binaryFeat[valid_ids], labels[valid_ids]
test_context, test_codesnippet, test_binaryFeat, test_labels = context[test_ids], codesnippet[test_ids], binaryFeat[test_ids], labels[test_ids]

dictionary = build_dict(numpy.concatenate([train_context, train_codesnippet]))
f = gzip.open(project + '.dict.pkl.gz', 'wb')
pkl.dump(dictionary, f, -1)
f.close()

train_t, train_d = grab_data(train_context, train_codesnippet, dictionary)
valid_t, valid_d = grab_data(valid_context, valid_codesnippet, dictionary)
test_t, test_d = grab_data(test_context, test_codesnippet, dictionary)

f = gzip.open(project + '.pkl.gz', 'wb')

pkl.dump((train_t, train_d,train_binaryFeat, train_labels,
              valid_t, valid_d,valid_binaryFeat, valid_labels,
              test_t, test_d,test_binaryFeat, test_labels), f, -1)
f.close()
