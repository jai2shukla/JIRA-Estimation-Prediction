from keras.layers import *
from keras.models import Model
from keras.constraints import *
from keras.regularizers import *
import gzip
import numpy
import cPickle

import load_data
import noise_dist
from NCE import *

arg = load_data.arg_passing(sys.argv)
dataset = '../data/' + arg['-data'] + '_pretrain.pkl.gz'
saving = arg['-saving']
emb_dim = arg['-dim']
max_len = arg['-len']
log = 'log/' + saving + '.txt'

n_noise = 100
print 'Loading data...'
train, valid, test = load_data.load(dataset)
valid = valid[-5000:]
vocab_size = arg['-vocab']

print 'vocab: ', vocab_size

######################################################
# prepare_lm load data and prepare input, output and then call the prepare_mask function
# all word idx is added with 1, 0 is for masking -> vocabulary += 1
train_x, train_y, train_mask = load_data.prepare_lm(train, vocab_size, max_len)
valid_x, valid_y, valid_mask = load_data.prepare_lm(valid, vocab_size, max_len)

print 'Data size: Train: %d, valid: %d' % (len(train_x), len(valid_x))

vocab_size += 1
n_samples, inp_len = train_x.shape

# Compute noise distribution and prepare labels for training data: next words from data + next words from noise
Pn = noise_dist.calc_dist(train, vocab_size)

labels = numpy.zeros((n_samples, inp_len, n_noise + 2), dtype='int64')
labels[:, :, 0] = train_mask
labels[:, :, 1] = 1

print 'Building model...'
# Build model
main_inp = Input(shape=(inp_len,), dtype='int64', name='main_inp')
next_inp = Input(shape=(inp_len,), dtype='int64', name='next_inp')

# Embed the context words to distributed vectors -> feed to GRU layer to compute the context vector
emb_vec = Embedding(output_dim=emb_dim, input_dim=vocab_size, input_length=inp_len,
                    #dropout=0.2,
                    mask_zero=True)(main_inp)

GRU_context = LSTM(input_dim=emb_dim, output_dim=emb_dim,
                   return_sequences=True)(emb_vec)
#GRU_context = Dropout(0.5)(GRU_context)

# feed output of GRU layer to NCE layer
nce_out = NCE_seq(input_dim=emb_dim, input_len=inp_len, vocab_size=vocab_size, n_noise=n_noise, Pn=Pn,
              )([GRU_context, next_inp])
nce_out_test = NCETest_seq(input_dim=emb_dim, input_len=inp_len, vocab_size=vocab_size)([GRU_context, next_inp])

# Call a model
model = Model(input=[main_inp, next_inp], output=[nce_out])
print model.summary()

optimizer = RMSprop(lr=0.02, rho=0.99, epsilon=1e-7) #optimizer = RMSprop(lr=0.01)
model.compile(optimizer=optimizer, loss=NCE_seq_loss)

testModel = Model(input=[main_inp, next_inp], output=[nce_out_test])
testModel.compile(optimizer='rmsprop', loss=NCE_seq_loss_test)

# save result to the filepath and wait if the result doesn't improve after 3 epochs, the lr will be divided by 2
fParams = 'bestModels/' + saving + '.hdf5'
callback = NCETestCallback(data=[valid_x, valid_y, valid_mask], testModel= testModel,
                           fResult=log, fParams=fParams)

json_string = model.to_json()
fModel = open('models/' + saving + '.json', 'w')
fModel.write(json_string)

print 'Training...'
his = model.fit([train_x, train_y], labels,
          batch_size=50, nb_epoch=20,
          callbacks=[callback])
