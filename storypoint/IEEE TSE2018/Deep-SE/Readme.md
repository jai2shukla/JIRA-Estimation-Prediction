# Deep-SE

## quick instruction

There are three folders:
1. data: contains the dataset in csv and the code for spliting data into training set, validation set, and test set
2. NCE: contains pretraining modules, LSTM modules, and LSTM feature extraction.
3. classification: contains the Highway Net.

--- step-by-step to run DEEP SE ---(LSTM+Highway)
1. put the csv files in /data
2. run command "python run_script.py" in /data to divide data and prepare dictionary
3. run command "python exp_lstm2v_pre.py" in /NCE for pretraining (this step takes very very long time!!!. It can be skipped since the model has been trained.)
4. run commnad "python exp_script.py" in /classification for running DEEP SE

The result is in /classification/log
<project name>_lstm_highway_dim<number of dimentsions>_reginphid_prefixed_lm_poolmean.txt
e.g. appceleratorstudio_lstm_highway_dim10_reginphid_prefixed_lm_poolmean.txt

To run the tool on your own data, you need to change the following to match with your filename:

| Folder         | File           | Variable     | Note                                                |
|----------------|----------------|--------------|-----------------------------------------------------|
| data           | run_script     | databaseDict | Pairs of dataset filename and its pretrain filename |
|                |                | dataPres     | Pretrain data file name                             |
| NCE            | exp_lstm2v_pre | dataPres     | Pretrain data file name                             |
| classification | exp_script     | databaseDict | Pairs of dataset filename and its pretrain filename |

## Dependencies
List of code dependencies and install command <br>
**beautifulsoup4**: `pip install BeautifulSoup4` <br>
**MySQLdb**: `pip install MySQLdb` <br>
**json**: `pip install json` <br>
**numpy**: `pip install numpy` <br>
**pandas**: `pip install pandas` <br>
**cPickle**: `pip install cpickle` <br>
**SciPy**: `pip install scipy` <br>
**scikit-learn**: `ppip install scikit-learn` <br>

Theano, keras 





--to be added--

list of hyperparameters for configuration
--to be added--

## setup instruction
## how to use
## configuration

# todo
 - add a proper story point report (issue key, des, estimated SP)
 - check all files, write a run script, remove password from data preprocessing, add classifier files

