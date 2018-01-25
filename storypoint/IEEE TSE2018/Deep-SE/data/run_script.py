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
    , 'moodle': 'moodle'
    , 'datamanagement': 'lsstcorp'
    , 'mule': 'mulesoft'
    , 'mulestudio': 'mulesoft'
    , 'springxd': 'spring'
    , 'talenddataquality': 'talendforge'
    , 'talendesb': 'talendforge'
}

dataPres = ['apache', 'appcelerator', 'duraspace', 'jira', 'moodle', 'lsstcorp', 'mulesoft', 'spring', 'talendforge']

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

for project, repo in datasetDict.items():
    print project + ' ' + repo
    cmd = 'python divide_data_sortdate.py ' + project
    print cmd
    os.system(cmd)

for dataPre in dataPres:
    print project + ' ' + repo
    cmd = 'python preprocess.py ' + dataPre
    print cmd
    os.system(cmd)

for project, repo in datasetDict.items():
    print project + ' ' + repo
    cmd = 'python preprocess_storypoint.py ' + project + ' ' + repo
    print cmd
    os.system(cmd)

