import os

model = 'lm'  # 'lm' if using lstm,
# 'bi' if using 'bilinear'
# 'w2v' if using 'word2vec'
# dims = ['100', '15', '20'] #, '15', '20']
dims = ['10', '50', '100', '200']
data = 'talendforge'
vocab = '5000'
maxlen = '100'

if model == 'lm':
    for dim in dims:
        command = 'python training_lm.py -data ' + data + \
                  ' -saving NCElm_' + data + '_dim' + dim + \
                  ' -vocab ' + vocab + ' -dim ' + dim + ' -len ' + maxlen
        print command
        os.system(command)

elif model == 'bi':
    for dim in dims:
        command = 'python training.py -data ' + data + \
                  ' -saving NCE_' + data + '_dim' + dim + \
                  ' -vocab ' + vocab + ' -dim ' + dim
        print command
        os.system(command)

else:
    for dim in dims:
        command = 'python doc2vec.py -data ' + data + \
                  ' -saving doc2vec_' + data + '_dim' + dim + \
                  ' -vocab ' + vocab + ' -dim ' + dim
        print command
        os.system(command)
