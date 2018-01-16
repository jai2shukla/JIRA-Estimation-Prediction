import csv
from sklearn import preprocessing
import MySQLdb
import sys
import numpy
from sklearn.preprocessing import MultiLabelBinarizer
from bs4 import BeautifulSoup

try:
    project = sys.argv[1]
except:
    print 'No argument'
    project = 'mesos'

print 'Project:' + project

connection = MySQLdb.connect(host='', user='', passwd='')
cursor = connection.cursor()

# query data
query = 'SELECT * FROM porru_dataset.' + project + ' ORDER BY openeddate ASC'
cursor.execute(query)
data = numpy.array(cursor.fetchall())

# train MultiLabelBinarizer for Component
query = 'SELECT DISTINCT SUBSTRING_INDEX(SUBSTRING_INDEX(t.components, \',\', n.n), \',\', -1) value ' \
        'FROM porru_dataset.' + project + ' t CROSS JOIN ' \
                                          '(' \
                                          'SELECT a.N + b.N * 10 + 1 n ' \
                                          'FROM ' \
                                          '(SELECT 0 AS N UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) a ' \
                                          ',(SELECT 0 AS N UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) b ' \
                                          'ORDER BY n ' \
                                          ') n ' \
                                          'WHERE n.n <= 1 + (LENGTH(t.components) - LENGTH(REPLACE(t.components, \',\', \'\'))) ' \
                                          'ORDER BY value'

cursor.execute(query)
componentList = numpy.array(cursor.fetchall())
mlb = MultiLabelBinarizer()
mlb.fit(componentList)
print 'No. of label: ' + str(len(mlb.classes_))
componentLabel = [None] * len(data)
for i in range(len(data)):
    componentLabel[i] = data[i, 5].split(',')
componentBinary = mlb.transform(componentLabel)

# train MultiLabelBinarizer for Type
query = 'SELECT DISTINCT type FROM porru_dataset.' + project
cursor.execute(query)
typeList = numpy.array(cursor.fetchall())
mlb_t = MultiLabelBinarizer()
mlb_t.fit(typeList)
print 'No. of label: ' + str(len(mlb_t.classes_))
typeLabel = [None] * len(data)
for i in range(len(data)):
    typeLabel[i] = data[i, 4].split(',')
typeBinary = mlb_t.transform(typeLabel)

# split code snippet
context = [None] * len(data)
codeSnippet = [None] * len(data)
for i in range(len(data)):
    description = BeautifulSoup(data[i][3].replace('{code}', '<code>'))
    summary = data[i][2]
    try:
        code = description.code.extract()
    except:
        code = ''
    context[i] = [summary + ' ' + str(description)]
    context[i][0] = context[i][0][:20000]
    codeSnippet[i] = [str(code)]
    codeSnippet[i][0] = codeSnippet[i][0][:20000]

processedData = numpy.concatenate([data[:, 0:2], context, codeSnippet, typeBinary, componentBinary], 1)

with open(project + '_porru.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['issuekey', 'storypoint', 'context',
                                           'codesnippet'] + mlb_t.classes_.tolist() + mlb.classes_.tolist(),
                            delimiter=',')
    writer.writeheader()
f.close()

with open(project + '_porru.csv', 'a') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(processedData)
f.close()
cursor.close()
connection.close()
