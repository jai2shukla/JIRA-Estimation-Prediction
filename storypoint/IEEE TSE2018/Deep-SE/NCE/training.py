from keras.layers import *
from keras.models import Model
from keras.constraints import *
from keras.regularizers import *

import numpy

import load_data
import noise_dist
from NCE import *


args = load_data.arg_passing(sys.argv)
dataset = '../data/' + args['-data'] + '_pretrain.pkl.gz'
saving = args['-saving']
emb_dim = args['-dim']
batch_size = 3000

n_context = 2
n_noise = 100

print 'Loading data...'
train, valid, test = load_data.load(dataset)

vocab_size = args['-vocab']
print 'vocab size: ', vocab_size
######################################################

train_x, train_y = load_data.prepare_NCE(train, n_context, vocab_size)
valid_x, valid_y = load_data.prepare_NCE(valid, n_context, vocab_size)
valid_x, valid_y = valid_x[:200000], valid_y[:200000]
test_x, test_y = load_data.prepare_NCE(test, n_context, vocab_size)

print 'Data size: Train: %d, valid: %d, test: %d' % (len(train_x), len(valid_x), len(test_x))

Pn = noise_dist.calc_dist(train, vocab_size)

labels = numpy.zeros((len(train_y), n_noise + 1), dtype='int64')
labels[:, 0] = 1

print 'Building model...'
# Build model
main_inp = Input(shape=(n_context,), dtype='int64', name='main_inp')
next_inp = Input(shape=(1,), dtype='int64', name='next_inp')

# Embed the context words to distributed vectors -> feed to NCEContext layer to compute the context vector
emb_vec = Embedding(output_dim=emb_dim, input_dim=vocab_size, input_length=n_context,
                    #W_regularizer=l1l2(l1=0.0005, l2=0.0005)
                    )(main_inp)

emb_context = NCEContext(input_dim=emb_dim, context_dim=n_context)(emb_vec)

nce_out = NCE(input_dim=emb_dim, vocab_size=vocab_size, n_noise=n_noise, Pn=Pn,
              )([emb_context, next_inp])
nce_out_test = NCETest(input_dim=emb_dim, vocab_size=vocab_size)([emb_context, next_inp])

model = Model(input=[main_inp, next_inp], output=nce_out)
print model.summary()

json_string = model.to_json()
fModel = open('models/' + saving + '.json', 'w')
fModel.write(json_string)

optimizer = RMSprop(lr=0.005)
#optimizer = Adam()
model.compile(optimizer=optimizer, loss=NCE_loss)

testModel = Model(input=[main_inp, next_inp], output=nce_out_test)
testModel.compile(optimizer=optimizer, loss=NCE_loss_test)

# save result to the filepath and wait if the result doesn't improve after 3 epochs, the lr will be divided by 2
callback = NCETestCallback(data=[valid_x, valid_y], testModel= testModel,
                           fResult='log/' + saving + '.txt', fParams='bestModels/' + saving + '.hdf5')

print 'Training...'
his = model.fit([train_x, train_y], labels,
          batch_size=batch_size, nb_epoch=100,
          callbacks=[callback])

