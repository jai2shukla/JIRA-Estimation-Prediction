import os
import glob
import numpy
import string

from subprocess import Popen, PIPE

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['C:/Perl64/perl/bin/perl', 'tokenizer.perl', '-l', 'en', '-q', '-']

def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    return toks

def remove_stops(sentences):
    f = open('stopwords.txt', 'r')
    stops = []
    for line in f:
        stops.append(line.split()[0])

    content = []
    for sen in sentences:
        words = sen.lower().split()
        ws = [w for w in words if w not in stops]
        content.append(' '.join(ws))

    return content

def load_authors():
    path = '../data/authors/'
    currdir = os.getcwd()

    def _get_authors():
        os.chdir(path + 'train')
        authors = []
        for author in glob.glob('*'):
            authors.append(author)

        os.chdir(currdir)
        return authors

    def _get_data(path, authors):
        sentences = []
        for i, author in enumerate(authors):
            for file in glob.glob(path + '/' + author + '/*'):
                f = open(file, 'r')
                for line in f:
                    s = line.split('\n')[0]
                    sentences.append(s)

        return sentences

    def _create_dataset(sentences):
        n_samples = len(sentences)
        idx_list = numpy.arange(n_samples, dtype='int32')
        numpy.random.shuffle(idx_list)

        n_train = int(n_samples * 2.0 / 3.0)
        n_valid = int(n_samples / 6.0)
        n_test = n_samples - n_train - n_valid
        train = []
        valid = []
        test = []

        for i in range(n_train):
            train.append(sentences[idx_list[i]])
        for i in range(n_valid):
            valid.append(sentences[idx_list[n_train + i]])
        for i in range(n_test):
            test.append(sentences[idx_list[n_train + n_valid + i]])

        return train, valid, test

    authors = _get_authors()
    sentences = _get_data(path + 'train', authors)
    sentences += _get_data(path + 'test', authors)
    sentences = tokenize(sentences)

    return _create_dataset(sentences)

def load_penn():
    path = 'E:/my_work/data/penntree/data/ptb.'

    def _load_text(file):
        f = open(file, 'r')
        sentences = []
        for line in f:
            sentences.append(line.split('\n')[0])

        return sentences

    train = _load_text(path + 'train.txt')
    valid = _load_text(path + 'valid.txt')
    test = _load_text(path + 'test.txt')

    return train, valid, test

def load_sent():
    def _create_dataset(sentences):
        n_samples = len(sentences)
        idx_list = numpy.arange(n_samples, dtype='int32')
        numpy.random.shuffle(idx_list)

        n_train = int(n_samples * 2.0 / 3.0)
        n_valid = int(n_samples / 6.0)
        n_test = n_samples - n_train - n_valid
        train = []
        valid = []
        test = []

        for i in range(n_train):
            train.append(sentences[idx_list[i]])
        for i in range(n_valid):
            valid.append(sentences[idx_list[n_train + i]])
        for i in range(n_test):
            test.append(sentences[idx_list[n_train + n_valid + i]])

        return train, valid, test

    def _remove_nonUnicode(sen):
        new_sen = ''
        words = sen.split()
        for w in words:
            isUnicode = True
            for c in w:
                if c not in string.ascii_letters:
                    isUnicode = False

            if isUnicode:
                new_sen += ' ' + w
        return new_sen

    path = 'E:/my_work/data/UCI/SentenceCorpus/unlabeled_articles'
    sentences = []
    for folder in glob.glob(path + '/*'):
        for file in glob.glob(folder + '/*'):
            f = open(file, 'r')
            for line in f:
                sen = line.split('\n')[0]
                sen = _remove_nonUnicode(sen)
                if len(sen) < 1: continue
                if sen[0] == '#': continue
                sentences.append(sen)

    sentences = tokenize(sentences)
    return _create_dataset(sentences)

def load(dataset='authors'):
    if dataset == 'authors':
        return load_authors()
    if dataset == 'sentences':
        return load_sent()
    return load_penn()

if __name__ == '__main__':
    load('sentences')