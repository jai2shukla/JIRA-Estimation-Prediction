import gensim
import load_data
import sys
import numpy
import cPickle
import gzip

arg = load_data.arg_passing(sys.argv)
dataset = '../data/' + arg['-data'] + '_pretrain.pkl.gz'
saving = arg['-saving']
emb_dim = arg['-dim']
vocab_size = arg['-vocab']

train, valid, test = load_data.load(dataset)
sentences = []
max_w = 0
count0 = 0
for sens in [train, valid, test]:
    for s in sens:
        w = []
        for i in s:
            w.append(str(i))
            max_w = max(max_w, i)
            if i == 0: count0 += 1
        sentences.append(w)

model = gensim.models.Word2Vec(sentences, min_count=20, size=emb_dim, window=3)

weight = numpy.zeros((vocab_size, emb_dim)).astype('float32')
for i in range(vocab_size):
    if str(i) in model:
        weight[i] = model[str(i)]

f = open('bestModels/' + saving + '.pkl', 'wb')
cPickle.dump((weight), f)