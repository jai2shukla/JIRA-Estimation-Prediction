# These are the datasets for our paper entitled "Predicting the delay of issues with due dates in software projects".

*[1] M. Choetkiertikul, H. K. Dam, T. Tran, and A. Ghose, “Predicting the delay of issues with due dates in software projects,” Empir. Softw. Eng., vol. 22, no. 3, pp. 1223–1263, 2017.*

### [preprint](https://github.com/SEAnalytics/datasets/blob/master/delayed%20issues/EMSE2017/preprint_EMSE2017.pdf)

```
@article{Choetkiertikul2017,
title = {{Predicting the delay of issues with due dates in software projects}},
author = {Choetkiertikul, Morakot and Dam, Hoa Khanh and Tran, Truyen and Ghose, Aditya},
doi = {10.1007/s10664-016-9496-7},
issn = {15737616},
journal = {Empirical Software Engineering},
number = {3},
pages = {1223--1263},
publisher = {Empirical Software Engineering},
volume = {22},
year = {2017}
}
```
***

**Description**

We collected data (past issues) from eight open source projects: Apache, Duraspace, Java.net, JBoss, JIRA, Moodle, Mulesoft, and WSO2. We extracted 19 risk factors (i.e. features) from issues. The feature extraction is corresponded with three prediction times: at the end of discussion time, at a time when a deadline (e.g. due date) was assigned to an issue, and at the creation time of an issue. The impact (i.e. dependent variable) is classified into three levles: high, medium, and low (including non-delay). 

<!-- **Source code**

We implemented our model on Matlab. The source code of the classifiers and the feature selections are provided. -->
