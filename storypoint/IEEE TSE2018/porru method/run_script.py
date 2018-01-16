import os

# This is a run script for Porru's method.
dataset = ['apstud', 'dnn', 'mesos', 'mule', 'nexus', 'timob', 'tistud', 'xd']

datasetPorru = ['apstud_porru', 'dnn_porru', 'mesos_porru', 'mule_porru', 'nexus_porru', 'timob_porru', 'tistud_porru', 'xd_porru']
# datasetPorru = ['timob_porru']


# run DatatoCSV.py
print 'Dataset to CSV'
for project in dataset:
    cmd = 'python DatatoCSV.py ' + project
    print cmd
    os.system(cmd)

# run DivideData.py
for project in dataset:
    cmd = 'python divide_data_sortdate.py ' + project
    print cmd
    os.system(cmd)

# run Porru's model
for project in datasetPorru:
    cmd = 'python preprocess.py ' + project
    print cmd
    os.system(cmd)

# run Porru's model
note = '4_porru_method'
for project in datasetPorru:
    cmd = 'python porrumethod.py ' + project + ' ' + project + '_porru_method'
    print cmd
    os.system(cmd)

    print 'compute error'
    cmd = 'python measurement.py -project ' + project + ' -fileName ' + project + '_porru_method' + ' -note ' + note
    os.system(cmd)
