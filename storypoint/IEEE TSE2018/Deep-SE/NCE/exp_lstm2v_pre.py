import os

mode = 'pretrain'

dims = ['10', '50', '100', '200']
dataPres = ['moodle', 'apache', 'appcelerator', 'duraspace', 'jira', 'lsstcorp', 'mulesoft', 'spring',
            'talendforge']  # pretrained data

vocab = '5000'

flag = ''

if mode == 'pretrain':
    for dim in dims:
        for dataPre in dataPres:
            command = flag + 'python lstm2vec_pretrain.py -data ' + dataPre + \
                      ' -saving lstm2v_' + dataPre + '_dim' + dim + ' -vocab ' + vocab + ' -dim ' + dim
            print command
            os.system(command)
