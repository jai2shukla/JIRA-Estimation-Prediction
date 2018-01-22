import os

mode = 'lstm2vec'  # mode = 'pretrain' for pretraining
# lstm2vec for lstm 2 vector

dims = ['10', '50', '100', '200']  # , '100'] #, '15', '20']
dataPre = 'moodle'  # pretrained data
datasets = []  # a list of datasets with story points (projects)
maxlen = '100'

datasetDict = {
    'mesos': 'apache',
    'usergrid': 'apache',
    'appceleratorstudio': 'appcelerator',
    'aptanastudio': 'appcelerator',
    'titanium': 'appcelerator',
    'duracloud': 'duraspace',
    'bamboo': 'jira',
    'clover': 'jira',
    'crucible': 'jira',
    'jirasoftware': 'jira',
    'moodle': 'moodle',
    'datamanagement': 'lsstcorp',
    'mule': 'mulesoft',
    'mulestudio': 'mulesoft',
    'springxd': 'spring',
    'talenddataquality': 'talendforge',
    'talendesb': 'talendforge'
}

vocab = '5000'  # vocab size. for large dataset, the vocab size should be set 5k - 10k.

flag = ''  # 'THEANO_FLAGS=\'floatX=float32,device=gpu\' '

if mode == 'pretrain':
    for dim in dims:
        command = flag + 'python lstm2vec_pretrain.py -data ' + dataPre + \
                  ' -saving lstm2v_' + dataPre + '_dim' + dim + \
                  ' -vocab ' + vocab + ' -dim ' + dim + ' -len ' + maxlen
        print command
        os.system(command)

elif mode == 'lstm2vec':
    for dim in dims:
        for project, repo in datasetDict.items():
            if repo == 'lsstcorp':
                vocab = '800'
            elif repo == 'mulesoft':
                vocab = '1600'
            else:
                vocab = '5000'
            command = flag + 'python lstm2vec.py -dataPre ' + repo + ' -data ' + project + \
                      ' -vocab ' + vocab + ' -dim ' + dim + ' -len ' + maxlen + \
                      ' -saving lstm2v_' + project + '_' + repo + '_dim' + dim
            print command
            os.system(command)
else:
    for dim in dims:
        for project, repo in datasetDict.items():
            command = 'python doc2vec.py -data ' + project + \
                      ' -saving doc2vec_' + project + '_' + repo + '_dim' + dim + \
                      ' -vocab ' + vocab + ' -dim ' + dim
            print command
            os.system(command)
