import cPickle
import numpy
import gzip
from keras.models import model_from_json
from NCE import *

def arg_passing(argv):
    # -data: dataset
    # -saving: log & model saving file
    # -dim: dimension of embedding
    # -reg: dropout: inp or hid or both

    i = 1
    arg_dict = {'-data': 'usergrid',
                '-dataPre': 'apache',
                '-saving': 'apache',
                '-seed': 1234,
                '-dim': 10,
                '-reg': '', # '', 'inp', 'hid' or 'inp_hid'
                '-seqM': 'lstm', # 'lstm', 'gru', 'rnn'
                '-nnetM': 'highway', #'dense', 'highway', 'resnet' - 'resnet' is not available now
                '-vocab': 500, # should be small
                '-pool': 'mean',
                '-ord': 0, # 0: categorical classification, 1: ordinal classification
                '-pretrain': 'x',
                '-len': 100
                }

    while i < len(argv) - 1:
        arg_dict[argv[i]] = argv[i+1]
        i += 2

    arg_dict['-dim'] = int(arg_dict['-dim'])
    arg_dict['-vocab'] = int(arg_dict['-vocab'])
    arg_dict['-seed'] = int(arg_dict['-seed'])
    arg_dict['-ord'] = int(arg_dict['-ord'])
    arg_dict['-len'] = int(arg_dict['-len'])
    return arg_dict


def load(path):
    f = gzip.open(path, 'rb')

    train_t, train_d, train_y, \
    valid_t, valid_d, valid_y, \
    test_t, test_d, test_y = cPickle.load(f)

    return train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y

def prepare_data(title, descr, vocab_size=1000, max_len=100):
    def create_mask(seqs):
        new_seqs = []
        for idx, s in enumerate(seqs):
            new_s = [w for w in s if w < vocab_size]
            if len(new_s) == 0: new_s = [0]
            new_seqs.append(new_s)

        seqs = new_seqs

        lengths = [min(max_len, len(s)) for s in seqs]
        maxlen = max_len
        n_samples = len(lengths)

        x = numpy.zeros((n_samples, maxlen)).astype('int64')
        mask = numpy.zeros((n_samples, maxlen)).astype('float32')

        for i, s in enumerate(seqs):
            l = lengths[i]
            mask[i, :l] = 1
            x[i, :l] = s[:l]
            x[i, :l] += 1

        return x, mask

    title, title_mask = create_mask(title)
    descr, descr_mask = create_mask(descr)

    return title, title_mask, descr, descr_mask

def load_weight(path):
    model_path = '../NCE/models/' + path + '.json'
    param_path = '../NCE/bestModels/' + path + '.hdf5'

    custom = {'NCEContext': NCEContext, 'NCE': NCE, 'NCE_seq': NCE_seq}
    fModel = open(model_path)
    model = model_from_json(fModel.read(), custom_objects=custom)
    model.load_weights(param_path)
    for layer in model.layers:
        weights = layer.get_weights()
        if 'embedding' in layer.name:
            return weights[0]

def load_w2v_weight(path):
    f = open('../NCE/bestModels/' + path, 'rb')
    return cPickle.load(f)

def to_features(list_seqs, emb_weight):
    vocab, dim = emb_weight.shape
    weight = numpy.zeros((vocab + 1, dim)).astype('float32')
    weight[1:] = emb_weight

    list_feats = []
    for seqs in list_seqs:
        n_samples, seq_len = seqs.shape
        feat = weight[seqs.flatten()].reshape([n_samples, seq_len, dim])
        list_feats.append(feat)
    return list_feats

def prepare_BoW(title, descr, vocab_size=1000):
    feats = numpy.zeros((len(title), vocab_size)).astype('float32')

    for seqs in [title, descr]:
        for i, s in enumerate(seqs):
            for word in s:
                if word < vocab_size:
                    feats[i, word] = 1

    return feats

def load_lstm2v_features(path):
    f = gzip.open(path, 'rb')
    train, train_y, valid, valid_y, test, test_y = cPickle.load(f)

    n_train = len(train) / 2
    n_valid = len(valid) / 2
    n_test  = len(test)  / 2
    train_x = numpy.concatenate([train[:n_train], train[n_train:]], axis=-1)
    valid_x = numpy.concatenate([valid[:n_valid], valid[n_valid:]], axis=-1)
    test_x  = numpy.concatenate([test[:n_test], test[n_test:]], axis=-1)

    return train_x, train_y, valid_x, valid_y, test_x, test_y

def load_doc2vec_features(path):
    f = gzip.open(path,'rb')
    train_x, train_y, valid_x, valid_y, test_x, test_y = cPickle.load(f)
    return train_x, train_y, valid_x, valid_y, test_x, test_y