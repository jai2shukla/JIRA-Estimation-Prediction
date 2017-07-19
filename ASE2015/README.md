# The datasets used in our paper entitled "Predicting delays in software projects using networked classification".

[1] M. Choetkiertikul, H. K. Dam, T. Tran, and A. Ghose, “Predicting delays in software projects using networked classification,” in Proceedings - 2015 30th IEEE/ACM International Conference on Automated Software Engineering, ASE 2015, 2016, pp. 353–364.

**Dataset**

Each dataset consists of task reports and linked data (explicit and implicit links).
1. A task in task reports consists of 3 JSON files:
  1. task details (<Task ID>.json),
  2. comments (<Task ID>_comment.json),
  3. change log (<Task ID>_CL.json)
Folder ‘summary’ contains the task summary of each task for applying Topic modelling techniques.

2. Linked data consists of several CSV files. One CSV file is one link type (<Link type>.csv). A link is presented by a pair of task IDs.