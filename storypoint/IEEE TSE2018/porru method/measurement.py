
import sys
import numpy
import MySQLdb
from sklearn.metrics import mean_absolute_error

def arg_passing_any(argv):
    i = 1
    arg_dict = {}
    while i < len(argv) - 1:
        arg_dict[argv[i]] = argv[i+1]
        i += 2
    return arg_dict

args = arg_passing_any(sys.argv)

try:
    project = args['-project']
    fileName = args['-fileName']
    note = args['-note']
except:
    print 'No args'
    project = 'mesos'
    fileName = 'mesos_porru_porru_method'
    note = 'test'

actualFile = 'log/output/' + fileName + '_actual.csv'
estimateFile = 'log/output/' + fileName + '_estimate.csv'


actual = numpy.genfromtxt(actualFile, delimiter=',')
estimate = numpy.genfromtxt(estimateFile, delimiter=',')

# Save absolute error in log/ar
ar_outputFileName = 'log/ar/' + fileName
numpy.savetxt(ar_outputFileName + ".csv", (numpy.absolute(actual - estimate)), delimiter=",", fmt='%1.4f')

MMRE = numpy.mean(2.0*((numpy.absolute(actual - estimate))/(actual + estimate)))
MAE = mean_absolute_error(actual, estimate)

print fileName + "," + note + "," + str(MMRE) + "," + str(MAE) + '\n'

with open('log/performance_all.csv', 'a') as myoutput:
    myoutput.write(fileName + "," + note + "," + str(MMRE) + "," + str(MAE) + '\n')