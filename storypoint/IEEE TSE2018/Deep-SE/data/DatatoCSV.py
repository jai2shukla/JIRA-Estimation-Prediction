import csv
from sklearn import preprocessing
import MySQLdb
import sys
import numpy
from sklearn.preprocessing import MultiLabelBinarizer
from bs4 import BeautifulSoup

try:
    project = sys.argv[1]
    repo = sys.argv[2]
except:
    print 'No argument'
    project = 'mesos'
    repo = 'porru_dataset'

print 'Project:' + project

def clean_sen(sen):
    sen = ''.join([c if ord(c) < 128 and ord(c) > 32 else ' ' for c in sen])
    return sen

connection = MySQLdb.connect(host='', user='', passwd='')
cursor = connection.cursor()

# query data
if project != 'pretrain':
    query = 'SELECT issuekey, substring(title,1,20000), substring(description,1,20000), storypoint FROM ' + repo + '.' + project + ' ORDER BY openeddate ASC'
    cursor.execute(query)
    data = numpy.array(cursor.fetchall())

    print 'No. of issue: ' + str(len(data))
    # clean data (remove special character)
    for i in range(len(data)):
        if data[i, 1] is None:
            data[i, 1] = 'None'
        else:
            data[i, 1] = clean_sen(data[i, 1])

    for i in range(len(data)):
        if data[i, 2] is None:
            data[i, 2] = 'None'
        else:
            data[i, 2] = clean_sen(data[i, 2])

    with open(project + '_' + 'porru' + '.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['issuekey', 'title', 'description', 'storypoint'], delimiter=',')
        writer.writeheader()
    f.close()

    with open(project + '_' + 'porru' + '.csv', 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)
    f.close()


if project == 'pretrain':
    query = 'SELECT issuekey, title, description, storypoint FROM ' + repo + '.storypoint_issue_pretrain'
    cursor.execute(query)
    data = numpy.array(cursor.fetchall())

    print 'No. of issue: ' + str(len(data))
    # clean data (remove special character)
    for i in range(len(data)):
        if data[i, 1] is None:
            data[i, 1] = 'None'
        else:
            data[i, 1] = clean_sen(data[i, 1])

    for i in range(len(data)):
        if data[i, 2] is None:
            data[i, 2] = 'None'
        else:
            data[i, 2] = clean_sen(data[i, 2])

    with open(repo + '_pretrain.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['issuekey', 'title', 'description', 'storypoint'], delimiter=',')
        writer.writeheader()
    f.close()

    with open(repo + '_pretrain.csv', 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)
    f.close()

cursor.close()
connection.close()