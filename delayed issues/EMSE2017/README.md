# These are the datasets and source code for our paper entitled "Predicting the delay of issues with due dates in software projects".

*[1] M. Choetkiertikul, H. K. Dam, T. Tran, and A. Ghose, “Predicting the delay of issues with due dates in software projects,” Empir. Softw. Eng., vol. 22, no. 3, pp. 1223–1263, 2017.*

**Dataset**

We collected data (past issues) from eight open source projects: Apache, Duraspace, Java.net, JBoss, JIRA, Moodle, Mulesoft, and WSO2. We extracted 19 risk factors (i.e. features) from issues. The feature extraction is corresponded with three prediction times: at the end of discussion time, at a time when a deadline (e.g. due date) was assigned to an issue, and at the creation time of an issue. The impact (i.e. dependent variable) is classified into three levles: high, medium, and low (including non-delay). 

**Source code**

We implemented our model on Matlab. The source code of the classifiers and the feature selections are provided.
