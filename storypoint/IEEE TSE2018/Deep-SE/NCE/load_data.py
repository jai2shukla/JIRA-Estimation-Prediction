import cPickle
import numpy
import theano
import theano.tensor as tensor
import gzip
import scipy.io as sio

def shared_data(data_xy, borrow=True):
    data_x, data_y = data_xy

    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, tensor.cast(shared_y, 'int32')

def load(path):
    f = gzip.open(path, 'rb')
    train, valid, test = cPickle.load(f)
    #print path, len(train[0]), len(valid[0])

    return train, valid, test

def load_data(path):
    f = gzip.open(path, 'rb')
    train_t, train_d, train_labels, valid_t, valid_d, valid_labels, test_t, test_d, test_labels = cPickle.load(f)

    train = train_t + train_d
    valid = valid_t + valid_d
    test = test_t + test_d

    return train, valid, test

def load_lstm2v(path):
    f = gzip.open(path, 'rb')
    train_t, train_d, train_labels, valid_t, valid_d, valid_labels, test_t, test_d, test_labels = cPickle.load(f)

    train = train_t + train_d
    valid = valid_t + valid_d
    test = test_t + test_d

    return train, train_labels, valid, valid_labels, test, test_labels

def prepare_NCE(seqs, n_context=2, vocab_size=10000, max_len=3000):
    new_seqs = []
    for i, s in enumerate(seqs):
        new_s = [w if w < vocab_size else 0 for w in s]
        new_seqs.append(new_s)
    seqs = new_seqs

    n_samples = 0
    for s in seqs: n_samples += max(0, min(max_len + 1, len(s)) - n_context)

    x = numpy.zeros((n_samples, n_context)).astype('int64')
    y = numpy.zeros((n_samples, 1)).astype('int64')

    idx = 0
    for s in seqs:
        for i in range(min(max_len + 1, len(s)) - n_context):
            x[idx] = s[i : i + n_context]
            y[idx] = s[i + n_context]

            idx += 1

    return x, y

def prepare_lm_test(seqs, vocab_size=10000, max_len=100):
    new_seqs = []
    for i, s in enumerate(seqs):
        new_s = [w if w < vocab_size else 0 for w in s]
        if len(new_s) < 1: new_s = [0]
        new_seqs.append(new_s)
    seqs = new_seqs

    lengths = [min(max_len, len(s)) for s in seqs]
    maxlen = max(lengths)
    n_samples = len(seqs)

    x = numpy.zeros((n_samples, maxlen)).astype('int64')
    mask = numpy.zeros((n_samples, maxlen)).astype('int64')

    for i, s in enumerate(seqs):
        l = lengths[i]
        mask[i, :l] = 1
        x[i, :l] = s[:l]
        x[i] += mask[i]

    return x, mask

def prepare_lm(seqs, vocab_size=10000, max_len=100):
    new_seqs = []
    for i, s in enumerate(seqs):
        new_s = [w if w < vocab_size else 0 for w in s]
        new_seqs.append(new_s)
    seqs = new_seqs

    lengths = [min(max_len, len(s)-1) for s in seqs]
    maxlen = max(lengths)
    n_samples = numpy.count_nonzero(lengths)

    x = numpy.zeros((n_samples, maxlen)).astype('int64')
    y = numpy.zeros((n_samples, maxlen)).astype('int64')
    mask = numpy.zeros((n_samples, maxlen)).astype('int64')

    idx = 0
    for i, s in enumerate(seqs):
        l = lengths[i]
        if l < 1: continue
        mask[idx, :l] = 1
        x[idx, :l] = s[:l]
        y[idx, :l] = s[1 : l+1]
        x[idx] += mask[idx]
        y[idx] += mask[idx]
        idx += 1

    return x, y, mask

def arg_passing(argv):
    # -data: dataset
    # -p: p
    # -gate: root or linear
    # -saving: log & model saving file
    # -dim: dimension of highway layers
    # -choice: large dataset or small one

    i = 1
    arg_dict = {'-data': 'authors',
                '-saving': 'highway',
                '-dim': 100,
                '-dataPre': 'apache',
                '-vocab': 5000,
                '-len': 100
                }

    while i < len(argv) - 1:
        arg_dict[argv[i]] = argv[i+1]
        i += 2

    arg_dict['-dim'] = int(arg_dict['-dim'])
    arg_dict['-vocab'] = int(arg_dict['-vocab'])
    arg_dict['-len'] = int(arg_dict['-len'])
    return arg_dict
