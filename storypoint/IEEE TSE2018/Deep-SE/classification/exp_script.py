import os
import sys

# Tuning parameters:
#mode = 'tunning'  # 'experiment'
mode = 'experiment'  # 'experiment'

model = 'seq'
dims = ['10', '50', '100', '200']  # , '15', '20']
# dims = ['200']  # , '15', '20']
# repository = 'apache'
# projects = ['usergrid'] #


datasetDict = {
    'mesos': 'apache',
    'usergrid': 'apache',
     'appceleratorstudio': 'appcelerator',
    'aptanastudio': 'appcelerator',
    'titanium': 'appcelerator',
    'duracloud': 'duraspace',
    'bamboo': 'jira',
    'clover': 'jira',
    'jirasoftware': 'jira',
    'crucible': 'jira',
    'moodle': 'moodle',
    'datamanagement': 'lsstcorp',
    'mule': 'mulesoft',
    'mulestudio': 'mulesoft',
    'springxd': 'spring',
    'talenddataquality': 'talendforge',
    'talendesb': 'talendforge'
}

nnet_models = ['highway']  # ['dense', 'highway']
seq_models = ['lstm']  # ['gru', 'lstm', 'rnn']
regs = ['inphid']  # ['x', 'inp', 'hid', 'inphid'] # 'x' means no dropout
pretrains = ['fixed_lm']  # ['x', 'finetune', 'fixed'] should use finetune_lm or fixed_lm for using lstm
# 'x' means no pretraining,
# 'finetune' means embedding matrix is initialized by pretrained parameters
# 'fixed' means using pretrained embedding matrix as input features
# add '_lm' if using 'lstm' for pretraining, default: 'bilinear' for pretraining
pools = ['mean']  # ['mean', 'max', 'last']
maxlen = '100'

# tunning parameter
# hdls = [2, 3, 5, 10, 20, 50, 100, 200]
hdls = [2, 3, 5, 10, 20, 30, 40, 50, 60, 80, 100, 200]
# hdls = [80,100,200]
# Running script:

if mode == 'experiment':
    if model == 'seq':
        for project, repository in datasetDict.items():
            for nnet in nnet_models:
                for seq in seq_models:
                    for dim in dims:
                        for reg in regs:
                            for pretrain in pretrains:
                                for pool in pools:
                                    cmd = 'python training.py -data ' + project + ' -dataPre ' + repository + \
                                          ' -nnetM ' + nnet + ' -seqM ' + seq + ' -dim ' + dim + \
                                          ' -reg ' + reg + ' -pretrain ' + pretrain + ' -pool ' + pool + ' -len ' + maxlen
                                    cmd += ' -saving ' + project + '_' + seq + '_' + nnet + '_dim' + dim + \
                                           '_reg' + reg + '_pre' + pretrain + '_pool' + pool
                                    # file name e.g. appceleratorstudio_lstm_highway_dim100_reginphid_prefixed_lm_poolmean.txt
                                    print cmd
                                    os.system(cmd)
elif mode == 'tunning':
    print 'tunning using MAE'
    if model == 'seq':
        for project, repository in datasetDict.items():
            for nnet in nnet_models:
                for seq in seq_models:
                    for dim in dims:
                        for reg in regs:
                            for pretrain in pretrains:
                                for pool in pools:
                                    for hdl in hdls:
                                        cmd = 'python training_tunning.py -data ' + project + ' -dataPre ' + repository + \
                                              ' -nnetM ' + nnet + ' -seqM ' + seq + ' -dim ' + dim + ' -len ' + maxlen + \
                                              ' -reg ' + reg + ' -pretrain ' + pretrain + ' -pool ' + pool + ' -hiddenLayer ' + str(
                                            hdl)
                                        cmd += ' -saving tune_' + project + '_dim_' + dim
                                        print cmd
                                        os.system(cmd)
#
# if model == 'BoW':
#     for s in project:
#         for nnet in nnet_models:
#             for dim in dims:
#                 for reg in regs:
#                     cmd = 'python baseline.py -data ' + s + ' -nnetM ' + nnet + ' -dim ' + dim + \
#                           ' -reg ' + reg + ' -ord 1'
#                     cmd += ' -saving BoW_' + s + '_' + nnet + '_dim' + dim + '_reg' + reg
#
#                     print cmd
#                     os.system(cmd)
