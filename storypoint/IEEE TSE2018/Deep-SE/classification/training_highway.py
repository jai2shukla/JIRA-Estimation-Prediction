__author__ = 'ptmin'

import numpy
import prepare_data
import sys

args = prepare_data.arg_passing(sys.argv)
numpy.random.seed(args['-seed'])

from keras.optimizers import *
from keras.objectives import *
from create_model import *

################################# LOAD DATA #######################################################
dim = args['-dim']
saving = args['-saving']
dropout = True if args['-reg'] != 'x' else False

dataset = '../NCE/data/lstm2v_' + args['-data'] + '_dim' + str(dim) + '.pkl.gz'
train_x, train_y, valid_x, valid_y, test_x, test_y = prepare_data.load_lstm2v_features(dataset)

print ('ntrain: %d, n_valid: %d, n_test: %d' % (len(train_y), len(valid_y), len(test_y)))

if train_y.dtype == 'float32':
    n_classes = -1
    loss = mean_squared_error
elif max(train_y) > 1:
    n_classes = max(train_y) + 1
    loss = sparse_categorical_crossentropy
else:
    n_classes = 1
    loss = binary_crossentropy

model = create_lstm2v(n_classes=n_classes, emb_dim=train_x.shape[-1], nnet_model='highway', dropout=True)

model.summary()
json_string = model.to_json()
fModel = open('models/' + saving + '.json', 'w')
fModel.write(json_string)

opt = RMSprop(lr=0.01)
model.compile(optimizer=opt, loss=loss)

train_y = numpy.expand_dims(train_y, -1)

fParams = 'bestModels/' + saving + '.hdf5'
#fResult = 'log/' + saving + '.txt'
fResult =  saving + '.txt'

if n_classes == -1: type = 'linear'
elif n_classes == 1: type = 'binary'
else: type = 'multi'

#saveResult = SaveResult([[valid_x], valid_y,
#                         [test_x], test_y],
#                        metric_type=type, fileResult=fResult, fileParams=fParams)


saveResult = SaveResult([[valid_x], valid_y,
                        [test_x], test_y],
                       metric_type=type, fileResult=fResult, fileParams=fParams, train_label=train_y)  # add mean of train_y


callbacks = [saveResult, NanStopping()]
his = model.fit([train_x], train_y,
                validation_data=([valid_x], numpy.expand_dims(valid_y, -1)),
                nb_epoch=1000, batch_size=100, callbacks=callbacks)