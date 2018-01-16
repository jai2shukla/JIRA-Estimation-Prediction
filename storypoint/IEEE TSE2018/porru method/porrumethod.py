import gzip
import cPickle
import numpy

import sys

from sklearn.feature_selection import chi2, f_classif

sys.path.append('PythonWorkspace/storypoint/Porrudataset/')

try:
    dataset = sys.argv[1] + '.pkl.gz'
    saving = sys.argv[2]
except:
    print 'No argument'
    dataset = 'PythonWorkspace/storypoint/Porrudataset/mesos_porru.pkl.gz'
    saving = 'test_porru_method'

f = gzip.open(dataset, 'rb')

train_context, train_code, train_binaryfeats, train_y, \
valid_context, valid_code, valid_binaryfeats, valid_y, \
test_context, test_code, test_binaryfeats, test_y = numpy.array(cPickle.load(f))


def listtostring(word_id):
    str_id = []
    for i in range(len(word_id)):
        str_id.append(' '.join(map(str, word_id[i])))
    return str_id


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())  # different object reference each time
    return list_of_objects


print 'convert word id to text...'

train_context = numpy.array(train_context)
train_code = numpy.array(train_code)
train_binaryfeats = numpy.array(train_binaryfeats)
train_y = numpy.array(train_y)

valid_context = numpy.array(valid_context)
valid_code = numpy.array(valid_code)
valid_binaryfeats = numpy.array(valid_binaryfeats)
valid_y = numpy.array(valid_y)

test_context = numpy.array(test_context)
test_code = numpy.array(test_code)
test_binaryfeats = numpy.array(test_binaryfeats)
test_y = numpy.array(test_y)

from sklearn.feature_extraction.text import TfidfVectorizer

# build TfidfVectorizer for monogram and bi-gram for contexts
tfidf_vectorizer = TfidfVectorizer(stop_words=None, lowercase=False, ngram_range=[1, 2])
tfidf_vectorizer.fit(listtostring(train_context))
train_context_tfidf = tfidf_vectorizer.transform(listtostring(train_context)).toarray()
valid_context_tfidf = tfidf_vectorizer.transform(listtostring(valid_context)).toarray()
test_context_tfidf = tfidf_vectorizer.transform(listtostring(test_context)).toarray()

# build TfidfVectorizer for monogram and bi-gram for codedsnippet
tfidf_vectorizer = TfidfVectorizer(stop_words=None, lowercase=False, ngram_range=[1, 2])
tfidf_vectorizer.fit(listtostring(train_code))
train_code_tfidf = tfidf_vectorizer.transform(listtostring(train_code)).toarray()
valid_code_tfidf = tfidf_vectorizer.transform(listtostring(valid_code)).toarray()
test_code_tfidf = tfidf_vectorizer.transform(listtostring(test_code)).toarray()

# concat all features

train_x = numpy.concatenate((train_context_tfidf, train_code_tfidf, train_binaryfeats), axis=1)
valid_x = numpy.concatenate((valid_context_tfidf, valid_code_tfidf, valid_binaryfeats), axis=1)
test_x = numpy.concatenate((test_context_tfidf, test_code_tfidf, test_binaryfeats), axis=1)


# train_x = numpy.concatenate((train_binaryfeats), axis=1)
# valid_x = numpy.concatenate((valid_binaryfeats), axis=1)
# test_x = numpy.concatenate((test_binaryfeats), axis=1)

# #############################################################################
# Create a feature-selection transform and an instance of SVM that we
# combine together to have an full-blown estimator

from sklearn.pipeline import Pipeline
from sklearn import svm, feature_selection

transform = feature_selection.SelectKBest(score_func=f_classif, k=50)
clf = Pipeline([('feat_select', transform), ('classifier', svm.SVC())])
clf.fit(train_x, numpy.floor(train_y))
predict = clf.predict(test_x)
print predict

numpy.savetxt('log/output/' + saving + "_actual.csv", test_y, delimiter=",")
numpy.savetxt('log/output/' + saving + "_estimate.csv", predict, delimiter=",")

# numpy.mean(2.0*((numpy.absolute(test_y - predict))/(test_y + predict)))
