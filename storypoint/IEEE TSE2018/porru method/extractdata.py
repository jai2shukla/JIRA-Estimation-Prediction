import json
import MySQLdb

connection = MySQLdb.connect(host='', user='', passwd='')
cursor = connection.cursor()

path = 'Replication_Package/dataset_open_source_15JUN2016/'

dataset = {
    # 'APSTUD__backup.json':'customfield_10003'
    # 'DNN__backup.json': 'customfield_10004'
        'MESOS__backup.json':'customfield_12310293'
    #     'MULE__backup.json':'customfield_10203'
    #     'TIMOB__backup.json':'customfield_10003'
    # 'TISTUD__backup.json':'customfield_10003'
    # 'XD__backup.json':'customfield_10142'
    # 'NEXUS__backup.json': 'customfield_10132'
    }

for filename, storypoint in dataset.items():
    with open(path + filename) as json_file:
        data = json.load(json_file)
    i = 0
    for item in data:
        fields = item['fields']
        changelog = item['changelog']
        histories = changelog['histories']
        check = True

        # filter1: storypoint changing
        for history in histories:
            his_items = history['items']
            for his_item in his_items:
                if his_item['field'] in [storypoint, 'storypoint']:
                    check = False

        # filter2: an issue must be addressed
        status = (fields['status'])['name']
        if status not in ['Closed', 'Resolved', 'Fixed', 'Completed', 'Done']:
            check = False

        # filter3: description, type, component changing
        for history in histories:
            his_items = history['items']
            for his_item in his_items:
                if his_item['field'] in ['description', 'issuetype', 'components', 'summary']:
                    check = False

        # filter4: planning poker card
        if fields[storypoint] not in [0, 0.5, 1, 2, 3, 5, 8, 13, 20, 40, 100]:
            if fields[storypoint] > 100:
                check = True
            else:
                check = False

        # all attributes must exist
        description = fields['description']
        if description in [None, '']:
            check = False

        summary = fields['summary']
        if summary == None:
            check = False

        components = fields['components']
        componentString = ''
        for component in components:
            componentString = componentString + ',' + component['name']
        componentString = componentString[1:]
        if componentString in [None, '']:
            check = False

        issuetype = (fields['issuetype'])['name']
        if issuetype == None:
            check = False

        openeddate = fields['created']
        sp_value = fields[storypoint]
        if check:
            # print item['key']
            # print issuetype
            # print componentString
            # print openeddate
            # print fields[storypoint]
            table = filename[:-13]
            i = i + 1
            try:
                cursor.execute('''INSERT into porru_dataset.''' + table + ''' (issuekey, storypoint, title, description, type, components, openeddate) VALUES
                               (%s, %s, %s, %s, %s, %s, date %s)''',
                               (item['key'], sp_value, summary, description, issuetype, componentString,
                                openeddate[:-18]))
                connection.commit()

            except:
                print 'skip'

    print i
    connection.close()
