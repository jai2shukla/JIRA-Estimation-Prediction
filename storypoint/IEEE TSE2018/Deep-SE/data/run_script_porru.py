import os

# run preprocess_storypoint.py

datasetDict = {
      'mesos': 'apache'
    , 'usergrid': 'apache'
    , 'appceleratorstudio': 'appcelerator'
    , 'aptanastudio': 'appcelerator'
    , 'titanium': 'appcelerator'
    , 'duracloud': 'duraspace'
    , 'bamboo': 'jira'
    , 'clover': 'jira'
    , 'jirasoftware': 'jira'
    , 'crucible':'jira'
    , 'moodle': 'moodle'
    , 'datamanagement': 'lsstcorp'
    , 'mule': 'mulesoft'
    , 'mulestudio': 'mulesoft'
    , 'springxd': 'spring'
    , 'talenddataquality': 'talendforge'
    , 'talendesb': 'talendforge'
}

datasetDict_PorruDB = {
      'apstud': 'porru_dataset'
    , 'dnn': 'porru_dataset'
    , 'mesos': 'porru_dataset'
    , 'mule': 'porru_dataset'
    , 'nexus': 'porru_dataset'
    , 'timob': 'porru_dataset'
    , 'tistud': 'porru_dataset'
    , 'xd': 'porru_dataset'
}

datasetDict_Porru = {
      'apstud_porru': 'appcelerator'
    , 'dnn_porru': 'dnn'
    , 'mesos_porru': 'apache'
    , 'mule_porru': 'mulesoft'
    , 'nexus_porru': 'sonatype'
    , 'timob_porru': 'appcelerator'
    , 'tistud_porru': 'appcelerator'
    , 'xd_porru': 'spring'
}


datasetPorru = ['apstud_porru', 'dnn_porru', 'mesos_porru', 'mule_porru', 'nexus_porru', 'timob_porru', 'tistud_porru', 'xd_porru']

datasetPorru_pretrain = ['sonatype']

for project, repo in datasetDict_PorruDB.items():
    print project + ' ' + repo
    cmd = 'python DatatoCSV.py ' + project + ' ' + repo
    print cmd
    os.system(cmd)
#
for repo in datasetPorru_pretrain:
    print repo
    cmd = 'python DatatoCSV.py ' + 'pretrain' + ' ' + repo
    print cmd
    os.system(cmd)
#
for project in datasetPorru:
    print project
    cmd = 'python divide_data_sortdate.py ' + project
    print cmd
    os.system(cmd)


for repo in datasetPorru_pretrain:
    print repo
    cmd = 'python preprocess.py ' + repo
    print cmd
    os.system(cmd)

for project, repo in datasetDict_Porru.items():
    print project
    cmd = 'python preprocess_storypoint.py ' + project + ' ' + repo
    print cmd
    os.system(cmd)



# for pretrain in pretrains:
#     for task in tasks:
#         if pretrain == 'x':
#             x = 'BoW_' + task
#             cmd = 'python classifier_br.py -data ' + task + ' -pretrain ' + pretrain + ' -saving RF_' + x
#             print cmd
#             os.system(cmd)
#         else:
#             x = 'lstm2v_' + task
#             for dim in dims:
#                 cmd = 'python classifier_br.py -data ' + task + ' -dim ' + dim + ' -pretrain ' + pretrain + ' -saving RF_' + x + '_dim' + dim
#                 print cmd
#                 os.system(cmd)
