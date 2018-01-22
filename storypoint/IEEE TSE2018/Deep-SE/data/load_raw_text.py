import pandas
import re
import numpy

def normalize(seqs):
    for i, s in enumerate(seqs):
        words = s.split()
        if len(words) < 1:
            seqs[i] = 'null'

    return seqs

def load_pretrain(path):
    data = pandas.read_csv(path).values
    return normalize(data[:, 1].astype('str')), normalize(data[:, 2].astype('str'))

def load(path):
    def cut_of90(labels):
        val_y = list(set(labels))
        val_y.sort()
        l_dict = dict()
        for i, val in enumerate(val_y): l_dict[int(val)] = i

        count_y = [0] * len(val_y)
        for label in labels:
            count_y[l_dict[int(label)]] += 1

        n_samples = len(labels)
        s, threshold = 0, 0
        for i, c in enumerate(count_y):
            s += c
            if s * 10 >= n_samples * 9:
                threshold = val_y[i]
                break
        for i, l in enumerate(labels):
            labels[i] = min(threshold, l)

        return labels.astype('float32')

    data = pandas.read_csv(path).values
    title = normalize(data[:, 1].astype('str'))
    description = normalize(data[:, 2].astype('str'))
    labels = data[:, 3].astype('float32')

    return title, description, cut_of90(labels)