% testing
setdbprefs('DataReturnFormat', 'dataset');
setdbprefs('NullNumberRead', 'NaN');
setdbprefs('NullStringRead', 'null');

conn = database('', 'ReshDBAdmin', '4688021', 'Vendor', 'MYSQL', 'Server', 'localhost', 'PortNumber', 3306);

projectName = {'Apache','Duraspace','Java.net','Jboss','JIRA','Moodle','Mulesoft','WSO2'};
databaseName = {'apache_ph2','duraspace_ph2','javanet','jboss_ph2','jira','moodle_ph2','mulesoft','wso2'};

columnNames = 'a.issuekey,impact,type,discussion,repetition,perofdelay,workload,priority,no_comment,no_priority_change,no_fixversion,no_fixversion_change,no_issuelink,no_blocking,no_blockedby,no_affectversion, reporterrep,no_des_change';
nonFeatures = {'issuekey','impact'};
catagoricalFeatures = {'type','priority'};
%%

for i=1:length(databaseName)
    fprintf('Project: %s',projectName{i});
    query = ['select ',columnNames,' from ', databaseName{i},'.issue_key_training a inner join ', databaseName{i},'.issue_feature b on a.issuekey = b.issuekey'];
    curs = exec(conn, query);    
    curs = fetch(curs);
    data = curs.Data;
    
    data(:,nonFeatures) = [];
    data(:,catagoricalFeatures) = [];
    collintest(data,'plot','on');
    
end