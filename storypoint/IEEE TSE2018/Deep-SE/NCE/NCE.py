import numpy
import theano
from theano import config
import theano.tensor as tensor
import theano.tensor.shared_randomstreams as RS

import keras
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import *

SEED = 1234

class NCEContext(Layer):
    def __init__(self, init='glorot_uniform', activation='linear',
                 weights=None, input_dim=None, context_dim=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=False, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights

        self.input_dim = input_dim
        self.context_dim = context_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)

        super(NCEContext, self).__init__(**kwargs)

    def build(self, input_shape): #input shape: nsamples * n_context * dim
        self.C = self.init((self.context_dim, self.input_dim),
                           name='{}_C'.format(self.name))
        self.trainable_weights = [self.C]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.input_dim)

    def call(self, x, mask=None):
        #x shape: nsamples * n_context * dim
        #out shape: nsamples * dim
        out = self.C[None, :, :] * x
        out = out.sum(axis=-2)
        return out

    def get_config(self):
        config = {'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim,
                  'context_dim': self.context_dim}
        base_config = super(NCEContext, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class NCE(Layer):
    def __init__(self, init='glorot_uniform', activation='linear',
                 input_dim=None, vocab_size=None, n_noise = 25, Pn=[0.5, 0.5],
                 weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.n_noise = n_noise
        self.Pn = theano.shared(numpy.array(Pn).astype(config.floatX))
        self.rng = RS.RandomStreams(seed=SEED)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)

        super(NCE, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.init((self.vocab_size, self.input_dim),
                           name='{}_W'.format(self.name))

        if self.bias:
            self.b = self.init((self.vocab_size,),
                               name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint

        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        return (None, self.n_noise + 1)

    def call(self, inputs, mask=None):
        context = inputs[0] #shape: n_samples * dim
        next_w = inputs[1] #shape: n_samles * 1

        n_samples = next_w.shape[0]
        n_next = self.n_noise + 1

        #generate n_noise samples from noise distribution Pn.
        noise_w = self.rng.choice(size=(n_samples, self.n_noise), a=self.Pn.shape[0], p=self.Pn)
        next_w = tensor.concatenate([next_w, noise_w], axis=-1)

        W_ = self.W[next_w.flatten()].flatten().reshape([n_samples, n_next, self.input_dim])
        b_ = self.b[next_w.flatten()].reshape([n_samples, n_next])

        # compute s_theta(w): scores of words under the model
        s_theta = (context[:, None, :] * W_).sum(axis=-1) + b_
        # compute the scores of words under the noise distribution: log(k * Pn(w))
        noiseP = self.Pn[next_w.flatten()].reshape([n_samples, n_next])
        noise_score = K.log(self.n_noise * noiseP)

        # the difference in the scores of words under the model and the noise distribution
        # shape: n_samples * n_next
        out = s_theta - noise_score

        return activations.sigmoid(out)

    def get_config(self):
        config = {'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim,
                  'vocab_size': self.vocab_size,
                  'n_noise': self.n_noise
                  }
        base_config = super(NCE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class NCE_seq(NCE):
    def __init__(self, input_len=10, **kwargs):
        self.input_len = input_len
        super(NCE_seq, self).__init__(**kwargs)

    def compute_mask(self, input, mask=None):
        return mask[0]

    def get_output_shape_for(self, input_shape):
        return (None, self.input_len, self.n_noise + 1)

    def call(self, inputs, mask=None):
        context = inputs[0] #shape: n_samples * n_steps * dim
        next_w = inputs[1] #shape: n_samles * n_steps

        n_samples, n_steps = next_w.shape
        n_next = self.n_noise + 1

        #generate n_noise samples from noise distribution Pn.
        noise_w = self.rng.choice(size=(n_samples, n_steps, self.n_noise), a=self.Pn.shape[0], p=self.Pn)
        next_w = next_w.flatten().reshape([n_samples, n_steps, 1])
        next_w = tensor.concatenate([next_w, noise_w], axis=-1) # shape: n_samples * n_steps * n_next

        W_ = self.W[next_w.flatten()].flatten().reshape([n_samples, n_steps, n_next, self.input_dim])
        b_ = self.b[next_w.flatten()].reshape([n_samples, n_steps, n_next])

        # compute s_theta(w): scores of words under the model
        # s_theta shape: n_samples * n_steps * n_next
        s_theta = (context[:, :, None, :] * W_).sum(axis=-1) + b_
        # compute the scores of words under the noise distribution: log(k * Pn(w))
        noiseP = self.Pn[next_w.flatten()].reshape([n_samples, n_steps, n_next])
        noise_score = K.log(self.n_noise * noiseP)

        # the difference in the scores of words under the model and the noise distribution
        # output shape: n_samples, n_steps, n_next
        out = s_theta - noise_score

        return activations.sigmoid(out)

class NCETest(NCE):
    def get_output_shape_for(self, input_shape):
        return (None, 1)

    def call(self, inputs, mask=None):
        context = inputs[0] # shape: n_samples * dim
        next_w = inputs[1] # shape: n_samples * 1
        n_samples = next_w.shape[0]

        out = K.dot(context, K.transpose(self.W)) + self.b
        out = activations.softmax(out)
        next_w = next_w.flatten()
        return out[tensor.arange(n_samples), next_w]

class NCETest_seq(NCETest):
    def __init__(self, input_len=10, **kwargs):
        self.input_len = input_len
        super(NCETest_seq, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (None, self.input_len)

    def compute_mask(self, input, mask=None):
        return mask[0]

    def call(self, inputs, mask=None):
        context = inputs[0] # shape: n_samples * n_steps * dim
        next_w = inputs[1] # shape: n_samples * n_steps
        n_samples, n_steps = next_w.shape
        vocab_size = self.W.shape[0]

        out = K.dot(context, K.transpose(self.W)) + self.b
        out = activations.softmax(out)
        out = out.flatten().reshape([n_samples*n_steps, vocab_size])
        next_w = next_w.flatten()

        prob = out[tensor.arange(n_samples*n_steps), next_w]
        prob = prob.reshape([n_samples, n_steps])
        return prob


class NCETestCallback(Callback):
    def __init__(self, data, testModel, fResult, fParams, patient=3):
        self.testModel = testModel
        self.fResult = fResult
        self.fParams = fParams
        self.patient = patient
        self.num_patient = patient
        self.best_epoch = 0
        self.best_loss = 100000.0

        self.do_test = False
        if len(data) == 2 or len(data) == 4:
            self.isSeq = 0
            self.valid_x = data[0]
            self.valid_y = data[1]
            self.valid_mask = data[0]

            if len(data) == 4:
                self.do_test = True
                self.test_x = data[2]
                self.test_y = data[3]
                self.test_mask = data[2]

        else:
            self.isSeq = 1
            self.valid_x, self.valid_y, self.valid_mask = data[0], data[1], data[2]
            if len(data) == 6:
                self.do_test = True
                self.test_x, self.test_y, self.test_mask = data[3], data[4], data[5]

        flog = open(fResult, 'w')
        flog.write('epoch\ttr_loss\tv_ppl\n')
        flog.close()
        super(NCETestCallback, self).__init__()

    def _compute_result(self, x, y, mask):
        y_pred = self.testModel.predict([x, y], batch_size=30)
        if self.isSeq:
            per = perplexity(mask, y_pred, 1)
        else:
            per = perplexity(mask, y_pred, 0)
        return per

    def on_epoch_end(self, epoch, logs={}):
        weights = self.model.get_weights()
        self.testModel.set_weights(weights)

        v_per = self._compute_result(self.valid_x, self.valid_y, self.valid_mask)
        if self.do_test:
            t_per = self._compute_result(self.test_x, self.test_y, self.test_mask)

        if self.best_loss < v_per:
            self.patient -= 1

            if self.patient == 0:
                lr = self.model.optimizer.lr / 2.0
                self.model.optimizer.lr = lr
                self.patient = self.num_patient
        else:
            self.patient = self.num_patient
            self.best_loss = v_per
            self.best_epoch = epoch
            self.model.save_weights(self.fParams, overwrite=True)

        print ('validation perplexity: %.4f' % v_per)

        train_loss = 0
        if 'loss' in logs:
            train_loss = logs['loss']

        f = open(self.fResult, 'a')
        f.write('%d\t%.4f\t%.4f' % (epoch, train_loss, v_per))
        if self.do_test:
            f.write('\t%.4f' % t_per)
        f.write('\tBest at epoch %d' % self.best_epoch)
        f.write('\n')
        f.close()

def NCE_seq_loss(y_true, y_pred):
    # y_true[:, :, 0]: masking matrix
    # y_true[:, :, 1] = 1: words from data
    # y_true[:, :, 2:] = 0: words from noise distribution
    # y_pred: probability of the word to be from data - shape: n_samples * n_steps * (n_noise + 1)

    loss = K.binary_crossentropy(y_pred, y_true[:, :, 1:])
    loss = loss.sum(axis=-1)
    loss *= y_true[:, :, 0] # masking matrix
    return K.sum(loss) / K.sum(y_true[:, :, 0])

def NCE_seq_loss_test(y_true, y_pred):
    # y_pred: n_samples * n_steps - probability of next word to be the corresponding word in y_true
    # y_true: masking matrix

    loss = -tensor.log(y_pred)
    loss *= y_true
    loss = K.sum(loss) / K.sum(y_true)
    return K.exp(loss)

def NCE_loss(y_true, y_pred): #n_samples * n_next
    loss = K.binary_crossentropy(y_pred, y_true)
    loss = K.mean(loss.sum(axis=-1))
    return loss

def NCE_loss_test(y_true, y_pred): #(n_samples,)
    loss = - tensor.log(y_pred)
    loss = K.mean(loss)
    loss = K.exp(loss)

    return loss

def perplexity(y_true, y_pred, isSeq = 0): #(n_samples,) or (n_samples, n_steps)
    eps = 1e-4
    loss = - numpy.log(y_pred + eps)

    if isSeq: # sequence
        loss *= y_true
        loss = numpy.sum(loss) / numpy.sum(y_true)
    else:
        loss = numpy.mean(loss)

    loss = numpy.exp(loss)
    return loss