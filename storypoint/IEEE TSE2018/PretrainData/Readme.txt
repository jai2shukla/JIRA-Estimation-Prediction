---Pre-train data---

Pre-training is a way to come up with a good parameter initialization without using the labels (i.e. ground-truth story points). We pre-train the lower layers of LD-RNN (i.e. embedding and LSTM), which operate at the word level.

We provided maximum 50,000 issues without story points in each repository for pre-training in csv format contained in PretrainData.zip. The csv file is named as "<repository name>_pretrain.csv" which consists of 4 columns: issue key, title, and description. Note that column "story point" contains null.

Pre-train data list
-------------------
apache_pretrain.csv                                         
appcelerator_pretrain.csv                                    
duraspace_pretrain.csv
moodle_pretrain.csv              
atlassian_pretrain.csv
lsstcorp_pretrain.csv           
mulesoft_pretrain.csv          
spring_pretrain.csv                
talendforge_pretrain.csv         
                     