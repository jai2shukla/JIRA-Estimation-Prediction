from keras.models import Model
from keras.layers import *
from sklearn import metrics
from keras.callbacks import *
import numpy

class SaveResult(Callback):
    '''
    Compute result after each epoch. Return a log of result
    Arguments:
        data_x, data_y, metrics
    '''

    def __init__(self, data=None, metric_type='binary', fileResult='', fileParams='', train_label='', minPatience=5,
                 maxPatience=30, hiddenLayer=0):
        # metric type can be: binary, multi
        super(SaveResult, self).__init__()

        self.valid_x = None
        self.valid_y = None
        self.test_x = None
        self.test_y = None
        self.do_test = False

        self.train_label = None
        self.actual = None
        self.estimate = None
        self.hdl = hiddenLayer
        # print 'Number of hiddenLayer' + str(self.hdl)
        f = open('log/' + fileResult, 'w')

        if 'binary' in metric_type:
            fm = ['auc', 'f1', 'pre', 'rec']
        elif 'multi' in metric_type:
            fm = ['ma_f1', 'mi_f1', 'pre', 'rec']
        else:
            fm = ['mae', 'sa', 'pr25', 'n/a']

        if len(data) >= 2:
            self.valid_x, self.valid_y = data[0], data[1]
            f.write('epoch\tloss\tv_loss\t|\tv_' + fm[0] + '\tv_' + fm[1] + '\tv_' + fm[2] + '\tv_' + fm[3] + '\t|')
        if len(data) == 4:
            self.test_x, self.test_y = data[2], data[3]
            f.write('\tt_' + fm[0] + '\tt_' + fm[1] + '\tt_' + fm[2] + '\tt_' + fm[3])
            self.do_test = True
        f.write('\n')
        f.close()

        self.bestResult = 1000.0 if 'linear' in metric_type else 0.0
        self.bestEpoch = 0
        # wait to divide the learning rate. if reach the maxPatience -> stop learning
        self.wait = 0
        self.patience = minPatience
        self.maxPatience = maxPatience

        self.metric_type = metric_type
        self.fileResult = fileResult
        self.fileParams = fileParams
        self.train_label = train_label

    def _compute_result(self, x, y_true, ):
        def pr25(y_true, y_pred):
            y_true = numpy.expand_dims(y_true, -1)
            mre = abs(y_true - y_pred) / y_true
            m = (mre <= 0.25).astype('float32')
            return 100.0 * numpy.sum(m) / len(y_true)

        y_pred = self.model.predict(x, batch_size=x[0].shape[0])
        if numpy.isnan(y_pred).any(): return 0.0, 0.0, 0.0, 0.0

        if 'binary' in self.metric_type:
            fp, tp, thresholds = metrics.roc_curve(y_true, y_pred)
            auc = metrics.auc(fp, tp)
            y_pred = numpy.round(y_pred)
            f1 = metrics.f1_score(y_true, y_pred)
            pre = metrics.precision_score(y_true, y_pred)
            rec = metrics.recall_score(y_true, y_pred)
        elif 'multi' in self.metric_type:
            y_pred = numpy.argmax(y_pred, axis=1)
            auc = metrics.f1_score(y_true, y_pred, average='macro')
            f1 = metrics.f1_score(y_true, y_pred, average='micro')
            pre = metrics.precision_score(y_true, y_pred, average='micro')
            rec = metrics.recall_score(y_true, y_pred, average='micro')
        else:
            auc = metrics.mean_absolute_error(y_true, y_pred)  # mae
            pre = pr25(y_true, y_pred)
            # pre, rec = auc, f1 # default

            y_pred_guess = numpy.ones(len(y_true)) + numpy.mean(self.train_label)
            mae_guess = metrics.mean_absolute_error(y_true, y_pred_guess)
            sa = (1 - (auc / mae_guess)) * 100
            f1 = sa
            rec = 0  # no use metric
            # self.ar = numpy.abs(numpy.subtract(y_true, y_pred))
            # self.ar = y_true
            # print y_true
            self.actual = y_true
            self.estimate = y_pred
        return auc, f1, pre, rec

    def better(self, a, b, sign):
        if sign == -1:
            return a < b
        else:
            return a > b

    def write_ar(self, fileResult, actual, estimate):
        ar_f = open('log/ar/' + fileResult, 'w')

        ar = numpy.abs(numpy.subtract(actual, estimate[:, 0]))
        for i in ar:
            ar_f.write(str(i))
            ar_f.write('\n')
        ar_f.close()
        raw = open('log/ar/raw_' + fileResult, 'w')
        diff = numpy.subtract(actual, estimate[:, 0])
        for i in diff:
            raw.write(str(i))
            raw.write('\n')
        raw.close()

    def write_mre(self, fileResult, actual, estimate):
        mre_f = open('log/mre/' + fileResult, 'w')

        mre = numpy.divide(numpy.abs(numpy.subtract(actual, estimate[:, 0])), actual)
        for i in mre:
            mre_f.write(str(i))
            mre_f.write('\n')
        mre_f.close()

    def save_result_tuning_hdl(self, epoch, result, hdl):
        path = 'log/tune_parameter/' + self.fileResult
        with open(path, 'a') as file:
            file.write(str(hdl) + '\t' + str(epoch) + '\t' + str(result) + '\n')
        file.close()

    def on_epoch_end(self, epoch, logs={}):
        v_auc, v_f1, v_pre, v_rec = self._compute_result(self.valid_x, self.valid_y)

        f = open('log/' + self.fileResult, 'a')
        f.write('%d\t%.4f\t%.4f\t|' % (epoch, logs['loss'], logs['val_loss']))
        f.write('\t%.4f\t%.4f\t%.4f\t%.4f\t|' % (v_auc, v_f1, v_pre, v_rec))
        if self.do_test:
            t_auc, t_f1, t_pre, t_rec = self._compute_result(self.test_x, self.test_y)
            f.write('\t%.4f\t%.4f\t%.4f\t%.4f\t|' % (t_auc, t_f1, t_pre, t_rec))

        # print '----' + self.metric_type
        if 'linear' in self.metric_type:
            compare_val = v_auc
            sign = -1
            # print (compare_val, self.bestResult, self.better(compare_val, self.bestResult, -1))
        else:
            compare_val = v_f1
            sign = 1

        if self.better(compare_val, self.bestResult, sign):
            self.bestResult = compare_val
            self.bestEpoch = epoch
            self.model.save_weights(self.fileParams, overwrite=True)
            self.wait = 0
            # save ar here
            self.write_ar(self.fileResult, self.actual, self.estimate)
            self.write_mre(self.fileResult, self.actual, self.estimate)
            #self.save_result_tuning_hdl(self.bestEpoch, self.bestResult, self.hdl)
        f.write('  Best result at epoch %d\n' % self.bestEpoch)
        f.close()

        if not self.better(compare_val, self.bestResult, sign):
            self.wait += 1
            if self.wait == self.patience:
                self.wait = 0
                self.patience += 5

                lr = self.model.optimizer.lr / 2.0
                self.model.optimizer.lr = lr
                if self.patience > self.maxPatience:
                    self.model.stop_training = True


class NanStopping(Callback):
    def __init__(self):
        super(NanStopping, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        for k in logs.values():
            if numpy.isnan(k):
                self.model.stop_training = True


class PoolingSeq(Layer):
    # pooling a sequence of vector to a vector
    # mode can be: mean, max or last (the last vector of the sequence)
    def __init__(self, mode='mean', **kwargs):
        self.mode = mode

        super(PoolingSeq, self).__init__(**kwargs)

    def compute_mask(self, input, mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (None, input_shape[0][-1])

    def call(self, inputs, mask=None):
        seqs = inputs[0]
        mask = inputs[1]

        if self.mode == 'mean':
            seqs = seqs * mask[:, :, None]
            pooled_state = K.sum(seqs, axis=1) / K.sum(mask, axis=1)[:, None]
        elif self.mode == 'max':
            seqs = seqs * mask[:, :, None]
            pooled_state = K.max(seqs, axis=1)
        else:  # self.mode = last
            pooled_state = seqs[:, -1]

        return pooled_state


def create_highway(below_layer, out_dim, hdl=10):
    shared_highway = Highway(activation='relu', init='glorot_normal', transform_bias=-1)
    hidd = below_layer
    print 'Number of hiddenLayer' + str(hdl)
    for i in range(
            hdl):  # it is the number of hidden layers in the highway net, you can change the range(10) to any number you like,as this is a hyper-parameter we need to specify
        hidd = shared_highway(hidd)

    return hidd


def create_dense(below_layer, out_dim):
    hidd = Dense(output_dim=out_dim, activation='relu', init='glorot_normal')(below_layer)

    return hidd


def create_lstm2v(n_classes, emb_dim, nnet_model='highway', dropout=True):
    hdim = int(emb_dim / 2)
    nnet_dict = {'highway': create_highway, 'dense': create_dense}
    if n_classes == -1:
        top_act = 'linear'
    elif n_classes == 1:
        top_act = 'sigmoid'
    else:
        top_act = 'softmax'

    inp = Input(shape=(emb_dim,), dtype='float32', name='input')
    hidd = Dense(output_dim=hdim, activation='relu', init='glorot_normal')(inp)
    if dropout: hidd = Dropout(0.5)(hidd)

    hidd = nnet_dict[nnet_model](hidd, hdim)
    if dropout: hidd = Dropout(0.5)(hidd)
    top_hidd = Dense(output_dim=abs(n_classes), activation=top_act)(hidd)
    model = Model(input=[inp], output=top_hidd)

    return model


def create_model(n_classes, vocab_size, inp_len, emb_dim,
                 seq_model='lstm', nnet_model='highway', pool_mode='mean',
                 dropout_inp=False, dropout_hid=True, emb_weight=None, hidden_layer=None):
    if emb_weight is not None:
        emb_weight = [emb_weight[:vocab_size]]
    seq_dict = {'lstm': LSTM, 'gru': GRU, 'rnn': SimpleRNN}
    nnet_dict = {'highway': create_highway, 'dense': create_dense}
    if n_classes == -1:
        top_act = 'linear'
    elif n_classes == 1:
        top_act = 'sigmoid'
    else:
        top_act = 'softmax'

    title_inp = Input(shape=(inp_len,), dtype='int64', name='title_inp')
    descr_inp = Input(shape=(inp_len,), dtype='int64', name='descr_inp')

    title_mask = Input(shape=(inp_len,), dtype='float32', name='title_mask')
    descr_mask = Input(shape=(inp_len,), dtype='float32', name='descr_mask')

    if dropout_inp:
        drop_rate = 0.2
    else:
        drop_rate = 0.0

    embedding = Embedding(output_dim=emb_dim, input_dim=vocab_size, input_length=inp_len,
                          mask_zero=True, weights=emb_weight,
                          dropout=drop_rate)
    seq_layer = seq_dict[seq_model](input_dim=emb_dim, output_dim=emb_dim,
                                    return_sequences=True, dropout_U=drop_rate, dropout_W=drop_rate)

    title_emb = embedding(title_inp)
    descr_emb = embedding(descr_inp)

    title_hid = seq_layer(title_emb)
    descr_hid = seq_layer(descr_emb)

    pooled_title = PoolingSeq(mode=pool_mode)([title_hid, title_mask])
    pooled_descr = PoolingSeq(mode=pool_mode)([descr_hid, descr_mask])

    hidd = merge([pooled_title, pooled_descr], mode='ave')
    if dropout_hid:
        hidd = Dropout(0.5)(hidd)

    hidd = nnet_dict[nnet_model](hidd, emb_dim, hidden_layer)
    hidd = Dropout(0.5)(hidd)
    top_hidd = Dense(output_dim=abs(n_classes), activation=top_act)(hidd)

    model = Model(input=[title_inp, title_mask, descr_inp, descr_mask], output=top_hidd)

    return model


def create_fixed(n_classes, inp_len, emb_dim,
                 seq_model='lstm', nnet_model='highway', pool_mode='mean',
                 dropout_inp=False, dropout_hid=True, hidden_layer=10):
    seq_dict = {'lstm': LSTM, 'gru': GRU, 'rnn': SimpleRNN}
    nnet_dict = {'highway': create_highway, 'dense': create_dense}
    if n_classes == -1:
        top_act = 'linear'
    elif n_classes == 1:
        top_act = 'sigmoid'
    else:
        top_act = 'softmax'

    title_inp = Input(shape=(inp_len, emb_dim), dtype='float32', name='title_inp')
    descr_inp = Input(shape=(inp_len, emb_dim), dtype='float32', name='descr_inp')

    title_mask = Input(shape=(inp_len,), dtype='float32', name='title_mask')
    descr_mask = Input(shape=(inp_len,), dtype='float32', name='descr_mask')

    if dropout_inp:
        drop_rate = 0.2
    else:
        drop_rate = 0.0

    seq_layer = seq_dict[seq_model](input_dim=emb_dim, output_dim=emb_dim,
                                    return_sequences=True, dropout_U=drop_rate, dropout_W=drop_rate)

    title_hid = seq_layer(title_inp)
    descr_hid = seq_layer(descr_inp)

    pooled_title = PoolingSeq(mode=pool_mode)([title_hid, title_mask])
    pooled_descr = PoolingSeq(mode=pool_mode)([descr_hid, descr_mask])

    hidd = merge([pooled_title, pooled_descr], mode='ave')
    if dropout_hid:
        hidd = Dropout(0.5)(hidd)

    hidd = nnet_dict[nnet_model](hidd, emb_dim, hidden_layer)
    hidd = Dropout(0.5)(hidd)
    top_hidd = Dense(output_dim=abs(n_classes), activation=top_act)(hidd)

    model = Model(input=[title_inp, title_mask, descr_inp, descr_mask], output=top_hidd)

    return model


def create_BoW(n_classes, vocab_size, hid_dim,
               nnet_model='highway', dropout=True):
    nnet_dict = {'highway': create_highway, 'dense': create_dense}
    if n_classes == -1:
        top_act = 'linear'
    elif n_classes == 1:
        top_act = 'sigmoid'
    else:
        top_act = 'softmax'

    input = Input(shape=(vocab_size,), dtype='float32', name='input')
    hidd = Dense(output_dim=hid_dim, activation='relu', init='glorot_normal')(input)
    hidd = Dropout(0.5)(hidd)
    hidd = nnet_dict[nnet_model](hidd, hid_dim)
    hidd = Dropout(0.5)(hidd)
    top_hidd = Dense(output_dim=abs(n_classes), activation=top_act)(hidd)

    model = Model(input=input, output=top_hidd)

    return model
