#The term schedule that gets displayed. Can do multiple terms in the case of displaying
#summer and fall at the same time. ie termNames ['2201','2208']
termNames=['2218']

majorTemplate='in/majorPage.html.mako'

#Add new majors here.
#Name: short name for the major
#classFile: the csv file containing all the classes in this majors curriculum
#asof: the date that the major curriculum was aquired
majors=[
     {'name': 'SNRE', 'classFile': 'majorClassLists/SNREList.csv', 'asof': 'Oct 10,2015'},
     {'name': 'WEC', 'classFile': 'majorClassLists/WECList.csv', 'asof': 'Oct 10,2015'}
]

#Add new semesters here.
#Name: The term code, see below. 
#prettyName: the more comprehendable name. eg. Fall 2015
#termSchedule: the filename for the downloaded csv file for the schedule. All should be semesterData/YYYYXX.csv
#
#The new API started being the sole source in spring 2020. With that term codes are:
#  CYYM, where C = 2, YY = the last 2 digits of the year, and M is 8 or 1 for fall or spring
#
#TODO: New codes for Summer. Its special since it has several mini-terms.
terms=[
     {'name' :'2218',   'prettyName':'Fall 2021', 'termSchedule': 'semesterData/fall2021.csv'}, 
     {'name' :'2211',   'prettyName':'Spring 2021', 'termSchedule': 'semesterData/spring2021.csv'}, 
     {'name' :'2208',   'prettyName':'Fall 2020', 'termSchedule': 'semesterData/fall2020.csv'}, 
     {'name' :'2201',   'prettyName':'Spring 2020', 'termSchedule': 'semesterData/spring2020.csv'}, 
     {'name' :'201908', 'prettyName':'Fall 2019', 'termSchedule': 'semesterData/201908.csv'}, 
     {'name' :'201906', 'prettyName':'Summer 2019', 'termSchedule': 'semesterData/201906.csv'}, 
     {'name' :'201901', 'prettyName':'Spring 2019', 'termSchedule': 'semesterData/201901.csv'}, 
     {'name' :'201808', 'prettyName':'Fall 2018', 'termSchedule': 'semesterData/201808.csv'}, 
     {'name' :'201806', 'prettyName':'Summer 2018', 'termSchedule': 'semesterData/201806.csv'}, 
     {'name' :'201801', 'prettyName':'Spring 2018', 'termSchedule': 'semesterData/201801.csv'}, 
     {'name' :'201708', 'prettyName':'Fall 2017', 'termSchedule': 'semesterData/201708.csv'}, 
     {'name' :'201706', 'prettyName':'Summer 2017', 'termSchedule': 'semesterData/201706.csv'}, 
     {'name' :'201701', 'prettyName':'Spring 2017', 'termSchedule': 'semesterData/201701.csv'}, 
     {'name' :'201608', 'prettyName':'Fall 2016', 'termSchedule': 'semesterData/201608.csv'}, 
     {'name' :'201606', 'prettyName':'Summer 2016', 'termSchedule': 'semesterData/201606.csv'}, 
     {'name' :'201601', 'prettyName':'Spring 2016', 'termSchedule': 'semesterData/201601.csv'}, 
     {'name' :'201508', 'prettyName':'Fall 2015', 'termSchedule': 'semesterData/201508.csv'}, 
     {'name' :'201506', 'prettyName':'Summer 2015', 'termSchedule': 'semesterData/201506.csv'}, 
     {'name' :'201501', 'prettyName':'Spring 2015', 'termSchedule': 'semesterData/201501.csv'}, 
     {'name' :'201408', 'prettyName':'Fall 2014', 'termSchedule': 'semesterData/201408.csv'}, 
     {'name' :'201406', 'prettyName':'Summer 2014', 'termSchedule': 'semesterData/201406.csv'}, 
     {'name' :'201401', 'prettyName':'Spring 2014', 'termSchedule': 'semesterData/201401.csv'}, 
     {'name' :'201308', 'prettyName':'Fall 2013', 'termSchedule': 'semesterData/201308.csv'}, 
     {'name' :'201301', 'prettyName':'Spring 2013', 'termSchedule': 'semesterData/201301.csv'}, 
     {'name' :'201208', 'prettyName':'Fall 2012', 'termSchedule': 'semesterData/201208.csv'}, 
     {'name' :'201201', 'prettyName':'Spring 2012', 'termSchedule': 'semesterData/201201.csv'}, 
     {'name' :'201108', 'prettyName':'Fall 2011', 'termSchedule': 'semesterData/201108.csv'}, 
     {'name' :'201101', 'prettyName':'Spring 2011', 'termSchedule': 'semesterData/201101.csv'}, 
     {'name' :'201008', 'prettyName':'Fall 2010', 'termSchedule': 'semesterData/201008.csv'} 
]

#To deal with 100's of special topic classes that may or may not be on the curriculum (and if not, still deserve
#to be considered), show *all* special topcis classes from a few relevant departments
relevantDepts=['BOT','ZOO','FAS','WIS','FOR','GEO','ENV']

#Exclude any classes with these titles. Designed for research credits which I don't need to have on the site
classTitleExclusions=['SUPERVISED RESEARCH','MASTERS RESEARCH','DOCTORAL RESEARCH','ADVANCED RESEARCH',
                      'SUPERVISED TEACHING','INDIVIDUAL WORK','INDIVIDUAL STUDIES','SPECIAL TOPICS']

#Every dept has 'Special Topic' codes that are not necessarily in the curriculum. 
#Since they all share the same course codes with things thare *are* in the curriclum, 
#all special topics are included.
#This list is to go and find them and mark them "special topics" to indicate the class might
#need prior approval. 
#Theres probably a better way to account for these. maybe scrape the grad catalog website
specialTopicClasses=['ZOO6927',
                     'WIS6934',
                     'SWS6932',
                     'ALS5932',
                     'AOM6932',
                     'AEC6932',
                     'STA6934',
                     'ANS6932',
                     'ENY6932',
                     'NEM6932',
                     'AEB6933',
                     'ABE6933',
                     'PHC6937',
                     'LAS6938',
                     'GEO6938',
                     'HOS6932',
                     'MCB6937',
                     'PBC6937',
                     'FAS6932',
                     'AGR6932',
                     'BOT6935',
                     'ANG6930',
                     'ENV6935',
                     'ENV6932',
                     'FOR6934',
                     'MAT6932',
                     'LAW6930',
                     'SYA7933',
                     'GEB6930',
                     'AFS6905',
                     'VME6934'
                     ]
