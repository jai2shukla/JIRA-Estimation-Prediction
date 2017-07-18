Dataset description 
-------------------
In each project and in each prediction time consists of three csv files: 
    1. the features of iterations named as <project>_iteration_<prediction time> (e.g. apache_iteration_30)
    2. the features of issues named as <project>_issue_<prediction time> (e.g. apache_issue_30)
    3. the list of issue links named as <project>_issuelink_<prediction time> (e.g. apache_issuelink_30)

Linking between an iteration with associated issues
---------------------------------------------------
There are two primary keys: (boardid and sprintid) which are used to join iterations and issues.
Iteration's primary keys: boardid and sprintid
Issue's primary keys: boardid, sprintid, and issuekey

List of issue links
-------------------
An issue link is provided as a pair of issue keys and link name.
For example, 'CB-3401', 'CB-3928', 'is blocked by' This shows that issue CB-3401 is blocked by issue CB-3928.




