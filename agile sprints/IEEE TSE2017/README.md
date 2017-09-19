
# These are the datasets used in our paper titled "Predicting Delivery Capability in Iterative Software Development".

[1] M. Choetkiertikul, H. K. Dam, T. Tran, A. Ghose, and J. Grundy, “Predicting Delivery Capability in Iterative Software Development,” IEEE Trans. Softw. Eng., vol. 14, no. 8, pp. 1–1, 2017.
[preprint](datasets/agile sprints/IEEE TSE2017/preprint_IEEETSE2017.pdf)

```
@article{Choetkiertikul2017,
title = {{Predicting Delivery Capability in Iterative Software Development}},
author = {Choetkiertikul, Morakot and Dam, Hoa Khanh and Tran, Truyen and Ghose, Aditya and Grundy, John},
doi = {10.1109/TSE.2017.2693989},
issn = {0098-5589},
journal = {IEEE Transactions on Software Engineering},
number = {8},
pages = {1--1},
volume = {14},
year = {2017}
}
```
***

Description
-----------

We provide our datasets: the iterations and the associated issues, and our source code: the feature aggregations and the randomised ensemble methods to build predictive models.

The dataset consists of the iterations and the associated issues collected from five open source projects: Apache, JBoss, JIRA, MongoDB, Spring. In each project we provide the features of iterations, the features of issues, and the list of issue links extracted from four different prediction times: 0%, 30%, 50%, and 80% of planning duration.

In each project and in each prediction time consists of three csv files: 
    1. the features of iterations named as project_iteration_prediction time (e.g. apache_iteration_30)
    2. the features of issues named as project_issue_prediction time (e.g. apache_issue_30)
    3. the list of issue links named as project_issuelink_prediction time (e.g. apache_issuelink_30)

Linking between an iteration with associated issues
---------------------------------------------------
There are two primary keys: (boardid and sprintid) which are used to join iterations and issues.

List of issue links
-------------------
An issue link is provided as a pair of issue keys and link name.
For example, 'CB-3401', 'CB-3928', 'is blocked by' This shows that issue CB-3401 is blocked by issue CB-3928.



<!-- **Source code**

Our MATLAB source code consists of two main components:(1) the feature aggregations (i.e. Statistical aggregation, Feature aggregation using Bag-of-Words, and Graph-based aggregation) , and (2) the three randomized ensemble methods (i.e. Random Forests, Stochastic Gradient Boosting Machines, and Deep Neural Networks with Dropouts) -->
