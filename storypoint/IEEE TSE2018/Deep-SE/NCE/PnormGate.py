import theano
import numpy
import cPickle
import time

import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.engine.topology import Layer

class HighwayPnorm(Layer):
    # trans_gate can be 'linear' or 'root'.
    # if linear: alpha1 = sigmoid()
    # if root:   alpha1 = sigmoid()^(1/p)
    def __init__(self, init='glorot_uniform',
                 transform_bias=-1, p_norm=1, scale=2.1, trans_gate='linear',
                 activation='relu', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.transform_bias = transform_bias
        self.activation = activations.get(activation)
        self.p_norm = p_norm
        self.scale = scale
        self.trans_gate = trans_gate

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(HighwayPnorm, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.init((input_dim, input_dim),
                           name='{}_W'.format(self.name))
        self.W_carry = self.init((input_dim, input_dim),
                                 name='{}_W_carry'.format(self.name))

        # learnable p_norm with weight matrix input_dim*1
        if self.p_norm == 0:
            self.W_p = self.init((input_dim, 1),
                                 name='{}_W_p'.format(self.name))

        # learnable p_norm with weight matrix input_dim*input_dim
        elif self.p_norm <= -1:
            self.W_p = self.init((input_dim,input_dim),
                                 name='{}_W_p'.format(self.name))

        if self.bias:
            self.b = K.zeros((input_dim,), name='{}_b'.format(self.name))
            # initialize with a vector of values `transform_bias`
            self.b_carry = K.variable(numpy.ones((input_dim,)) * self.transform_bias,
                                      name='{}_b_carry'.format(self.name))

            # learnable p_norm with bias of 1 element
            if self.p_norm == 0:
                self.b_p = K.zeros((1,), name='{}_b_p'.format(self.name))

            # learnable p_norm with bias input_dim*1
            elif self.p_norm <= -1:
                self.b_p = K.zeros((input_dim,), name='{}_b_p'.format(self.name))

            if self.p_norm <= 0:
                self.trainable_weights = [self.W, self.b, self.W_carry, self.b_carry, self.W_p, self.b_p]
            else:
                self.trainable_weights = [self.W, self.b, self.W_carry, self.b_carry]
        else:
            if self.p_norm <= 0:
                self.trainable_weights = [self.W, self.W_carry, self.W_p]
            else:
                self.trainable_weights = [self.W, self.W_carry]

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
            self.constraints[self.W_carry] = self.W_constraint
            if self.p_norm <= 0: self.constraints[self.W_p] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint
            self.constraints[self.b_carry] = self.b_constraint
            if self.p_norm <= 0: self.constraints[self.b_p] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _compute_root(self, x): #gate: nsamples * dim
        if self.p_norm > 0:
            return self.p_norm

        u = K.dot(x, self.W_p)
        if self.bias:
            u += self.b_p
        root = activations.sigmoid(u)
        root = root * (self.scale - 1) + 1

        return root

    def call(self, x, mask=None):
        y = K.dot(x, self.W_carry)
        if self.bias:
            y += self.b_carry

        root = self._compute_root(x)

        transform_gate = activations.sigmoid(y)

        if self.trans_gate == 'linear':
            carry_gate = 1. - transform_gate ** root
        else:
            carry_gate = 1. - transform_gate
            transform_gate = transform_gate ** (1.0 / root)

        carry_gate = carry_gate ** (1.0 / root)

        y = K.dot(x, self.W)
        if self.bias:
            y += self.b
        act = self.activation(y)

        output = transform_gate* act + carry_gate * x
        return output

    def get_config(self):
        config = {'init': self.init.__name__,
                  'transform_bias': self.transform_bias,
                  'activation': self.activation.__name__,
                  'p_norm': self.p_norm,
                  'scale': self.scale,
                  'trans_gate': self.trans_gate,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(HighwayPnorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class GRUPnorm(Recurrent):
    def __init__(self, output_dim, p_norm=1, transform_bias=-1, scale=2.1, trans_gate='linear',
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.p_norm = p_norm
        self.scale = scale
        self.trans_gate = trans_gate
        self.transform_bias = transform_bias
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(GRUPnorm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        if self.p_norm <= -1:
            self.W_p = self.init((self.output_dim, self.output_dim),
                                 name='{}_W_p'.format(self.name))

            self.U_p = self.init((self.output_dim, self.output_dim),
                                 name='{}_U_p'.format(self.name))

            self.b_p = self.init((self.output_dim,),
                                 name='{}_b_p'.format(self.name))

        if self.consume_less == 'gpu':

            self.W = self.init((self.input_dim, 3 * self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 3 * self.output_dim),
                                     name='{}_U'.format(self.name))

            self.b = K.variable(numpy.hstack((numpy.zeros(self.output_dim),
                                           numpy.zeros(self.output_dim),
                                           numpy.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))

            if self.p_norm <= -1:
                self.trainable_weights = [self.W, self.U, self.b, self.W_p, self.U_p, self.b_p]
            else:
                self.trainable_weights = [self.W, self.U, self.b]
        else:

            self.W_z = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_z'.format(self.name))
            self.U_z = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_z'.format(self.name))
            self.b_z = K.variable(numpy.ones((self.output_dim,)) * self.transform_bias,
                                      name='{}_b_carry'.format(self.name))

            self.W_r = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_r'.format(self.name))
            self.U_r = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_r'.format(self.name))
            self.b_r = K.zeros((self.output_dim,), name='{}_b_r'.format(self.name))

            self.W_h = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_h'.format(self.name))
            self.U_h = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_h'.format(self.name))
            self.b_h = K.zeros((self.output_dim,), name='{}_b_h'.format(self.name))

            if self.p_norm <= -1:
                self.trainable_weights = [self.W_z, self.U_z, self.b_z,
                                          self.W_r, self.U_r, self.b_r,
                                          self.W_h, self.U_h, self.b_h,
                                          self.W_p, self.U_p, self.b_p]
            else:
                self.trainable_weights = [self.W_z, self.U_z, self.b_z,
                                          self.W_r, self.U_r, self.b_r,
                                          self.W_h, self.U_h, self.b_h]

            self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
            self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        numpy.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_z = time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            return K.concatenate([x_z, x_r, x_h], axis=2)
        else:
            return x

    def _compute_root(self, x, h): #gate: nsamples * dim
        if self.p_norm > 0:
            return self.p_norm

        u = K.dot(x, self.W_p) + K.dot(h, self.U_p) + self.b_p

        root = activations.sigmoid(u)
        root = root * (self.scale - 1) + 1

        return root


    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        if self.consume_less == 'gpu':

            matrix_x = K.dot(x * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

            x_z = matrix_x[:, :self.output_dim]
            x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.output_dim:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.consume_less == 'cpu':
                x_z = x[:, :self.output_dim]
                x_r = x[:, self.output_dim: 2 * self.output_dim]
                x_h = x[:, 2 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(x * B_W[2], self.W_h) + self.b_h
            else:
                raise Exception('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))


        root = self._compute_root(x, h_tm1)
        transform_gate = z

        if self.trans_gate == 'linear':
            carry_gate = 1. - transform_gate ** root
        else:
            carry_gate = 1. - transform_gate
            transform_gate = transform_gate ** (1.0 / root)

        carry_gate = carry_gate ** (1.0 / root)

        h = transform_gate * h_tm1 + carry_gate * hh
        return h, [h]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(3)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'p_norm': self.p_norm,
                  'scale': self.scale,
                  'trans_gate': self.trans_gate,
                  'transform_bias': self.transform_bias,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(GRUPnorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Conv2DHighwayPnorm(Layer):
    def __init__(self, nb_filter, nb_row, nb_col,
                 transform_bias=-1, p_norm=1,
                 init='glorot_uniform', activation='relu', weights=None,
                 border_mode='same', subsample=(1, 1), dim_ordering='th',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.transform_bias = transform_bias
        self.p_norm = p_norm
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(Conv2DHighwayPnorm, self).__init__(**kwargs)

    def setP(self, new_p):
        self.p_norm = new_p

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.W_gate = self.init(self.W_shape, name='{}_W_carry'.format(self.name))

        if self.bias:
            self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
            self.b_gate = K.variable(numpy.ones(self.nb_filter,), name='{}_b_gate'.format(self.name))
            self.trainable_weights = [self.W, self.b, self.W_gate, self.b_gate]
        else:
            self.trainable_weights = [self.W, self.W_gate]
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
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        # compute the candidate hidden state
        transform = K.conv2d(x, self.W, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          filter_shape=self.W_shape)
        if self.bias:
            if self.dim_ordering == 'th':
                transform += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                transform += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        transform = self.activation(transform)

        transform_gate = K.conv2d(x, self.W_gate, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          filter_shape=self.W_shape)
        if self.bias:
            if self.dim_ordering == 'th':
                transform_gate += K.reshape(self.b_gate, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                transform_gate += K.reshape(self.b_gate, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        transform_gate = K.sigmoid(transform_gate)

        carry_gate = 1.0 - transform_gate
        transform_gate **= (1.0 / self.p_norm)
        carry_gate **= (1.0 / self.p_norm)

        return transform * transform_gate + x * carry_gate

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'transform_bias': self.transform_bias,
                  'p_norm': self.p_norm,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(Conv2DHighwayPnorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
