from keras.layers import *
from keras.models import Model
from keras.constraints import *
from keras.regularizers import *
from keras.models import model_from_json
import gzip
import numpy
import cPickle

import load_data
import noise_dist
from NCE import *

arg = load_data.arg_passing(sys.argv)
emb_dim = arg['-dim']
data_pretrain = 'lstm2v_' + arg['-dataPre'] + '_dim' + str(emb_dim)
dataset = '../data/' + arg['-data'] + '.pkl.gz'
saving = arg['-saving']
max_len = arg['-len']

vocab_size = arg['-vocab']

print 'vocab: ', vocab_size

# save result to the filepath and wait if the result doesn't improve after 3 epochs, the lr will be divided by 2
model_path = 'models/' + data_pretrain + '.json'
param_path = 'bestModels/' + data_pretrain + '.hdf5'

custom = {'NCEContext': NCEContext, 'NCE': NCE, 'NCE_seq': NCE_seq}
fModel = open(model_path)
model = model_from_json(fModel.read(), custom_objects=custom)
model.load_weights(param_path)

# end pretraining.
# to feature
train, train_labels, valid, valid_labels, \
    test, test_labels = load_data.load_lstm2v(dataset)

train_x, train_mask = load_data.prepare_lm_test(train, vocab_size, max_len)
valid_x, valid_mask = load_data.prepare_lm_test(valid, vocab_size, max_len)
test_x, test_mask = load_data.prepare_lm_test(test, vocab_size, max_len)
get_lstm_output = K.function([model.layers[0].input],
                             [model.layers[2].output])

def lstm2feature(vecs, mask):
    # vecs: n_samples * n_steps * emb_dim
    # mask: n_samples * n_steps
    feats = vecs * mask[:, :, None]
    feats = numpy.sum(feats, axis=1) / numpy.sum(mask, axis=1)[:, None]
    return feats

train_lstm = get_lstm_output([train_x])[0]
valid_lstm = get_lstm_output([valid_x])[0]
test_lstm = get_lstm_output([test_x])[0]

train_feats = lstm2feature(train_lstm, train_mask)
valid_feats = lstm2feature(valid_lstm, valid_mask)
test_feats = lstm2feature(test_lstm, test_mask)

f = gzip.open('data/' + saving + '.pkl.gz', 'wb')
cPickle.dump((train_feats, train_labels, valid_feats, valid_labels, test_feats, test_labels), f)
f.close()