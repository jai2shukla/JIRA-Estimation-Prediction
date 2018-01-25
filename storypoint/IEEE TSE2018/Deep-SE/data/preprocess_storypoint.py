
import gzip
import sys
import cPickle
import load_raw_text
import preprocess

def main():
# load training data:
    data_path = sys.argv[1] + '.csv'
    title, description, labels = load_raw_text.load(data_path)

    f = open(sys.argv[1] + '_3sets.txt', 'r')
    train_ids, valid_ids, test_ids = [], [], []
    count = -2
    for line in f:
        if count == -2:
            count += 1
            continue

        count += 1
        ls = line.split()
        if ls[0] == '1': train_ids.append(count)
        if ls[1] == '1': valid_ids.append(count)
        if ls[2] == '1': test_ids.append(count)

    print 'ntrain, nvalid, ntest: ', len(train_ids), len(valid_ids), len(test_ids)

    train_title, train_description, train_labels = title[train_ids], description[train_ids], labels[train_ids]
    valid_title, valid_description, valid_labels = title[valid_ids], description[valid_ids], labels[valid_ids]
    test_title, test_description, test_labels = title[test_ids], description[test_ids], labels[test_ids]

    f_dict = gzip.open(sys.argv[2] + '.dict.pkl.gz', 'rb')
    dictionary = cPickle.load(f_dict)
    train_t, train_d = preprocess.grab_data(train_title, train_description, dictionary)
    valid_t, valid_d = preprocess.grab_data(valid_title, valid_description, dictionary)
    test_t, test_d = preprocess.grab_data(test_title, test_description, dictionary)

    f = gzip.open(sys.argv[1] + '.pkl.gz', 'wb')
    cPickle.dump((train_t, train_d, train_labels,
              valid_t, valid_d, valid_labels,
              test_t, test_d, test_labels), f, -1)
    f.close()

if __name__ == '__main__':
    main()