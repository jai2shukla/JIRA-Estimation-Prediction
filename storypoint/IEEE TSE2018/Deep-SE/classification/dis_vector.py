import prepare_data
import gzip
import cPickle

from sklearn.manifold import TSNE
from numpy import vstack, array

from scipy.cluster.vq import kmeans, vq

repo = 'mulesoft'
dim = '10'


def filter(word):
    if word.isalpha():
        return True
    else:
        return False


# load embedding vector
emb_weight = prepare_data.load_weight('lstm2v_' + repo + '_dim' + dim)
W = emb_weight[1:]
print 'Size of vector:'
print W.shape

# load dict
f = gzip.open('../data/' + repo + '.dict.pkl.gz', 'rb')
wordDict = cPickle.load(f)

print len(wordDict)
i = 1
for word, ids in sorted(wordDict.items(), key=lambda t: t[1]):
    print ids
    print word
    i += 1
    if i > 20:
        break

model = TSNE(n_components=2, random_state=0)
embed_W = model.fit_transform(W)
print embed_W
print embed_W.shape

outputFile = open('embed_weight/' + repo + '.txt', 'w')
# all
# centroids, _ = kmeans(embed_W, 10)
# idx, _ = vq(embed_W, centroids)
#
# # save to text file

# i = 0
# for word, ids in sorted(wordDict.items(), key=lambda t: t[1]):
#     if filter(word):
#         outputFile.write(
#             str(ids) + '\t' + word + '\t' + str(idx[i]) + '\t' + str(embed_W[i][0]) + '\t' + str(embed_W[i][1]) + '\n')
#     i += 1
#     if i >= 5000:
#         break
# outputFile.close()

# study on the top-500 highest frequency workds

wordList_f = {}
embed_W_f = [[0 for i in range(2)] for j in range(1000)]
i = 0
for word, ids in sorted(wordDict.items(), key=lambda t: t[1]):
    if filter(word):
        wordList_f[word] = ids
        print ids
        print i
        print embed_W[int(ids) - 1][0]
        print embed_W[int(ids) - 1][1]
        print '---'

        embed_W_f[i][0] = embed_W[int(ids) - 1][0]
        embed_W_f[i][1] = embed_W[int(ids) - 1][1]
        i += 1
        if i >= 500:
            break

centroids, _ = kmeans(embed_W_f, 10)
idx, _ = vq(embed_W_f, centroids)
i = 0
for word, ids in wordList_f.items():
    outputFile.write(
        str(ids) + '\t' + str(word) + '\t' + str(idx[i]) + '\t' + str(embed_W_f[i][0]) + '\t' + str(embed_W_f[i][1]) + '\n')
    i += 1
outputFile.close()
