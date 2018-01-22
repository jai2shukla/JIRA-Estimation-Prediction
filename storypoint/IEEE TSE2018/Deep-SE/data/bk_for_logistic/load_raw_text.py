import pandas
import numpy

def load(path):
    def normalize(seqs):
        for i, s in enumerate(seqs):
            if len(s) < 1:
                seqs[i] = 'null'
        return seqs

    def rare(labels):
        y_dict = dict()
        for label in labels:
            if label in y_dict:
                y_dict[label] += 1
            else: y_dict[label] = 1

        keys = y_dict.keys()
        rare_y = []
        for key in keys:
            if y_dict[key] < 20:
                rare_y.append(key)

        return rare_y


    data = pandas.read_csv(path).values
    title = normalize(data[:, 1].astype('str'))
    description = normalize(data[:, 2].astype('str'))
    labels = data[:, 3].astype('int64')

    return title, description, labels, rare(labels)

#load('data/apache_text.csv')