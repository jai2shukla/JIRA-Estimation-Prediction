import numpy
import prepare_data
import sys

import numpy
import prepare_data
import sys
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor

args = prepare_data.arg_passing(sys.argv)
numpy.random.seed(args['-seed'])
dim = args['-dim']

################################# LOAD DATA #######################################################
saving = args['-saving']
pretrain = args['-pretrain']
vocab_size = args['-vocab']
if pretrain == 'x':
    dataset = '../data/' + args['-data'] + '.pkl.gz'
    train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y = prepare_data.load(dataset)
    train_x = prepare_data.prepare_BoW(train_t, train_d, vocab_size)
    valid_x = prepare_data.prepare_BoW(valid_t, valid_d, vocab_size)
    test_x = prepare_data.prepare_BoW(test_t, test_d, vocab_size)
elif pretrain == 'doc2vec':
    dataset = '../NCE/data/' + pretrain + '_' + args['-data'] + '_dim' + str(dim) + '.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = prepare_data.load_lstm2v_features(dataset)
else:
    dataset = '../NCE/bestmodels/' + pretrain + '_' + args['-data'] + '_dim' + str(dim) + '.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = prepare_data.load_lstm2v_features(dataset)

print ('ntrain: %d, n_valid: %d, n_test: %d' % (len(train_y), len(valid_y), len(test_y)))

def evaluation(y_true, y_pred, y_guess): # morakot: added SA as a new metric by accept mean of train_y as MAEguess
    def pr25(y_true, y_pred):
        mre = abs(y_true - y_pred) / y_true
        m = (mre <= 0.25).astype('float32')
        return 100.0 * numpy.sum(m) / len(y_true)

    mae = metrics.mean_absolute_error(y_true, y_pred)
    pr25 = pr25(y_true, y_pred)
    y_pred_guess = numpy.ones(len(y_true)) + y_guess
    mae_guess = metrics.mean_absolute_error(y_true, y_pred_guess)
    sa = (1-(mae/mae_guess))*100

    return [mae, sa, pr25]

def seek_results(params):
    clf = RandomForestRegressor(n_estimators=params['n_estimators'],
             max_depth=params['max_depth'],
             min_samples_split=params['min_samples_split'])

    clf.fit(train_x, train_y)

    train_pred = clf.predict(train_x)
    valid_pred = clf.predict(valid_x)
    test_pred = clf.predict(test_x)


    train_res, valid_res, test_res  = evaluation(train_y, train_pred, numpy.mean(train_y)),\
                                      evaluation(valid_y, valid_pred, numpy.mean(train_y)), evaluation(test_y, test_pred, numpy.mean(train_y))
    # print train_res, valid_res, test_res
    ar = numpy.abs(numpy.subtract(test_y, test_pred)) # find absolute error i.e. |actual - estimate|, ar is an array
    return valid_res, test_res, ar # add return ar

def write_log(params, valid_res, test_res):
    f = open('log/' + saving + '.txt', 'a')
    for param in params:
        f.write(str(params[param]) + '\t')
    for res in [valid_res, test_res]:
        f.write('|')
        for r in res:
            f.write('%.4f\t' % r)

    f.write('\n')
    f.close()

def write_ar(ar):
    f = open('log/ar/' + saving + '.txt', 'w')
    for i in ar:
        f.write(str(i))
        f.write('\n')
    f.close()

params = dict()
best_valid, best_test= [10000, 0], [10000, 0]
for i in range(10):
    for max_d in [15]:
        for min_split in [1]:
            params['n_estimators'] = 50 + i * 5
            params['max_depth'] = max_d
            params['min_samples_split'] = min_split

# params['n_estimators'] = 50
# params['max_depth'] = 2
# params['min_samples_split'] = 4

            valid_res, test_res, ar = seek_results(params)
            write_log(params, valid_res, test_res)

            if valid_res[0] < best_valid[0]:
                best_valid = valid_res
                best_test = test_res
                write_ar(ar)

write_log(params, best_valid, best_test)