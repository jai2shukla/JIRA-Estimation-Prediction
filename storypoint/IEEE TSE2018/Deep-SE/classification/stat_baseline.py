# this script generate AR of mean and median for statistical camprison

import numpy
import gzip
import cPickle
import sys
import random
from sklearn import metrics

data = sys.argv[1]
baseline = sys.argv[2]


def load(path):
    f = gzip.open(path, 'rb')

    train_t, train_d, train_y, \
    valid_t, valid_d, valid_y, \
    test_t, test_d, test_y = cPickle.load(f)

    return train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y


def pr25(y_true, y_pred):
    mre = abs(y_true - y_pred) / y_true
    m = (mre <= 0.25).astype('float32')
    return 100.0 * numpy.sum(m) / len(y_true)


# load file
dataset = '../data/' + data + '.pkl.gz'
train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y = load(dataset)
print ('ntrain: %d, n_valid: %d, n_test: %d' % (len(train_y), len(valid_y), len(test_y)))

# cal estimate from mean of y_train
if baseline == 'mean':
    base_y = numpy.mean(train_y)
    y_pred_base = numpy.ones(len(test_y))*base_y
elif baseline == 'median':
    base_y = numpy.median(train_y)
    y_pred_base = numpy.ones(len(test_y))*base_y
elif baseline == 'random':
    y_pred_base = numpy.ones(len(test_y))
    for i in range(len(y_pred_base)):
        y_pred_base[i] = random.choice(train_y)
# print base_y


y_pred_guess = numpy.ones(len(test_y)) + numpy.mean(train_y)
mae = metrics.mean_absolute_error(test_y, y_pred_base)
mae_guess = metrics.mean_absolute_error(test_y, y_pred_guess)  # the result on the paper, MAE guess is calculated separately.
sa = (1 - (mae / mae_guess)) * 100
pred25 = pr25(test_y, y_pred_base)

f = open('log/baseline_stat/' + data + '_' + baseline + '.txt', 'w')
f.write('MAE\tSA\tPred(25)\n')
f.write(str(mae) + '\t' + str(sa) + '\t' + str(pred25))
f.write('\n')
f.close()
# save AR file
ar_f = open('log/baseline_stat/ar/' + data + '_' + baseline + '.txt', 'w')
ar = numpy.abs(numpy.subtract(test_y, y_pred_base))
#print ar
for i in ar:
    ar_f.write(str(i))
    ar_f.write('\n')
ar_f.close()


