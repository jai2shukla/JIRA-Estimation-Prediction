Please visit my Github repository for the dataset and source code: https://github.com/morakotch/soft-analytics



These are the datasets and supplementary resources for our submission entitled:

Predicting Delivery Capability in Iterative Software Development

We provide our datasets: the iterations and the associated issues, and our source code: the feature aggregations and the randomised ensemble methods to build predictive models. The detailed descriptions are provided in "readme.txt" in each section.



Datasets:

The dataset consists of the iterations and the associated issues collected from five open source projects: Apache, JBoss, JIRA, MongoDB, Spring. In each project we provide the features of iterations, the features of issues, and the list of issue links extracted from four different prediction times: 0%, 30%, 50%, and 80% of planning duration. Please see "readme.txt" for the detail.

Readme

Apache

JBoss

JIRA

MongoDB

Spring

Source code:

Our MATLAB source code consists of two main components:(1) the feature aggregations (i.e. Statistical aggregation, Feature aggregation using Bag-of-Words, and Graph-based aggregation) , and (2) the three randomized ensemble methods (i.e. Random Forests, Stochastic Gradient Boosting Machines, and Deep Neural Networks with Dropouts)

code