# gensim modules
import gzip

import cPickle
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from collections import namedtuple

import numpy
import load_data
from NCE import *

arg = load_data.arg_passing(sys.argv)
emb_dim = arg['-dim']
#data_pretrain = 'lstm2v_' + arg['-dataPre'] + '_dim' + str(emb_dim)
dataset = '../data/' + arg['-data'] + '.pkl.gz'
saving = arg['-saving']
max_len = arg['-len']

vocab_size = arg['-vocab']

# print emb_dim
# print dataset
# print saving
# print max_len
# print vocab_size

print 'vocab: ', vocab_size

train, train_labels, valid, valid_labels, test, test_labels = load_data.load_lstm2v(dataset)

analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')

def trainDoc2vec(documents):
    taggeddocuments = []
    for i, text in enumerate(documents):
        #words = str(text)
        words = ",".join(str(c) for c in text)
        words = words.split(',')
        # print words
        tags = [i]
        taggeddocuments.append(analyzedDocument(words, tags))
    model = Doc2Vec(taggeddocuments, size = emb_dim, window = 50, min_count = 1, workers = 4)
    return model

def Doc2vecModeltoFeat(model):
    feat = []
    for i in range(len(model.docvecs)):
        feat.append(model.docvecs[i])
    return feat

trainVec = trainDoc2vec(train)
validVec = trainDoc2vec(valid)
testVec = trainDoc2vec(test)

trainFeat = Doc2vecModeltoFeat(trainVec)
validFeat = Doc2vecModeltoFeat(validVec)
testFeat = Doc2vecModeltoFeat(testVec)

train_feats = numpy.array(trainFeat)
valid_feats = numpy.array(validFeat)
test_feats = numpy.array(testFeat)

train_labels = numpy.array(train_labels)
valid_labels = numpy.array(valid_labels)
test_labels = numpy.array(test_labels)

print saving
f = gzip.open('data/' + saving + '.pkl.gz', 'wb')
cPickle.dump((train_feats, train_labels, valid_feats, valid_labels, test_feats, test_labels), f)
f.close()

