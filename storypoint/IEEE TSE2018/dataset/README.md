# The datasets used in our paper entitled "Predicting delays in software projects using networked classification"

[1] M. Choetkiertikul, H. K. Dam, T. Tran, and A. Ghose, “Predicting delays in software projects using networked classification,” in Proceedings of the 30th IEEE/ACM International Conference on Automated Software Engineering, ASE 2015, 2016, pp. 353–364.

### [preprint](https://github.com/SEAnalytics/datasets/blob/master/delayed%20issues/ASE2015/preprint_ASE2015.pdf)

```
@inproceedings{Choetkiertikul2015,
title = {{Predicting delays in software projects using networked classification}},
author = {Choetkiertikul, Morakot and Dam, Hoa Khanh and Tran, Truyen and Ghose, Aditya},
booktitle = {Proceedings of the 30th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
doi = {10.1109/ASE.2015.55},
isbn = {9781509000241},
pages = {353--364},
year = {2015}
}
```
***

**Description**

Each dataset consists of task reports and linked data (explicit and implicit links).
1. A task in task reports consists of 3 JSON files:
  - task details (Task ID.json),
  - comments (Task ID_comment.json),
  - change log (Task ID_CL.json)

2. Folder ‘summary’ contains the task summary of each task for applying Topic modelling techniques.

3. Linked data consists of several CSV files. One CSV file is one link type (Link type.csv). A link is presented by a pair of task IDs.