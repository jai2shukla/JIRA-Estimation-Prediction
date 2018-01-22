#
# divide the dataset into 3 parts: training, validation and testing.
# Each line in output contains 3 numbers indicate the set (training, validation or testing) that the datapoint belongs.

import sys
from sklearn.cross_validation import StratifiedKFold
import pandas
import numpy

data_path = sys.argv[1] + '.csv'
data = pandas.read_csv(data_path).values
labels = data[:, 3].astype('int64')

skf = StratifiedKFold(labels, 3, shuffle=True)
for train_ids, nontrain_ids in skf:
    break

nontrain_labels = labels[nontrain_ids]
skf_nontrain = StratifiedKFold(nontrain_labels, 2, shuffle=True)
for valid, test in skf_nontrain:
    valid_ids = nontrain_ids[valid]
    test_ids = nontrain_ids[test]

divided_set = numpy.zeros((len(labels), 3)).astype('int64')

for i, ids in enumerate([train_ids, valid_ids, test_ids]):
    for idx in ids:
        divided_set[idx, i] = 1

f = open(sys.argv[1] + '_3sets.txt', 'w')
f.write('train\tvalid\ttest')
for s in divided_set:
    f.write('\n%d\t%d\t%d' % (s[0], s[1], s[2]))

f.close()