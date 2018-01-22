import os

# run preprocess_storypoint.py

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
    'moodle': 'moodle',
    'datamanagement': 'lsstcorp',
    'mule': 'mulesoft',
    'mulestudio': 'mulesoft',
    'springxd': 'spring',
    'talenddataquality': 'talendforge',
    'talendesb': 'talendforge',
}

dataPres = ['apache', 'appcelerator', 'duraspace', 'jira', 'moodle', 'lsstcorp', 'mulesoft', 'spring', 'talendforge']

for project, repo in datasetDict.items():
    print project + ' ' + repo
    cmd = 'python divide_data_sortdate.py ' + project
    print cmd
    os.system(cmd)

for project, repo in datasetDict.items():
    print project + ' ' + repo
    cmd = 'python preprocess_storypoint.py ' + project + ' ' + repo
    print cmd
    os.system(cmd)

for dataPre in dataPres:
    print project + ' ' + repo
    cmd = 'python preprocess.py ' + dataPre
    print cmd
    os.system(cmd)
