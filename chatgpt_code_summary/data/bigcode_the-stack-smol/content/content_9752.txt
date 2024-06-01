#!/usr/bin/python

import sys
import cgi
import cgitb
import sqlite3
reload(sys)
sys.setdefaultencoding('utf-8')
cgitb.enable()

# html
print("Content-type: text/html\n")
print('<meta charset="utf-8">')
print("<html><head>")
print('''<script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>''')
print("<title>BRITE REU Candidates</title>")
print('''<link rel="stylesheet" href="https://bioed.bu.edu/students_21/group_proj/group_K/css/nav.css">
         <link rel="stylesheet" href="https://bioed.bu.edu/students_21/group_proj/group_K/css/appadmin.css">
</head>''')

print("<body>")
print('''<div id="bg-image">''')
print('''<div id="topnav">
  <a href="https://bioed.bu.edu/cgi-bin/students_21/group_proj/group_K/show_applicant_admin.py">Applicant List</a>
  <a href="https://bioed.bu.edu/cgi-bin/students_21/group_proj/group_K/stats_admin.py">Applicant Statistics</a>
  <a href="#assign users">Assign Users</a>
  <a href="https://bioed.bu.edu/cgi-bin/students_21/group_proj/group_K/input_projects.py">Input Faculty Projects</a>
  <a href="https://bioed.bu.edu/cgi-bin/students_21/group_proj/group_K/review_summary_admin.py">View All Past Reviews</a>
  <a class="active" href="https://bioed.bu.edu/cgi-bin/students_21/group_proj/group_K/assign_candidate.py">Assign Candidates to Faculty</a>
  <a href="https://bioed.bu.edu/cgi-bin/students_21/group_proj/group_K/can_pref.py">Candidate Preferences</a>
  <a href="https://bioed.bu.edu/cgi-bin/students_21/group_proj/group_K/match.py">Match Candidates to Faculty</a>
  <a href="https://bioed.bu.edu/cgi-bin/students_21/group_proj/group_K/finalmatch.py">Final Matches</a>
  <a href="https://bioed.bu.edu/cgi-bin/students_21/group_proj/group_K/help_admin.py">Help</a>
  <a href="https://bioed.bu.edu/cgi-bin/students_21/group_proj/group_K/about_admin.py">About/Contact</a>

</div>''')

print("<h3>Select Checkboxes to Assign Candidates to Faculty Members</h3>") 
print("<h4>To Remove an Assignment, Uncheck the Checkbox</h4>")

#query to get candidate data for the rows
query1 = "SELECT cid, firstname, lastname FROM Applicant join Candidate on Applicant.aid=Candidate.cid;"

#query to get the faculty and project names for the table headers
query2 = 'SELECT pid, uid, fname || " " || lname || ":\n" || project_name FROM Project JOIN User using(uid) ORDER BY(lname);'

#query to get all current candidate-faculty pairs in the database
query3 = 'SELECT cid || "_" || pid, assigned_at FROM Assignment ORDER BY(cid);'

#start connection
connection = sqlite3.connect('db/BRITEREU.db')
c = connection.cursor()
try:
    #execute query 1 
    c.execute(query1)
    #get results to above standard query
    results1 = c.fetchall()
except Exception:
    print("<p><font color=red><b>Error Query 1</b></font></p>")

try:
    #execute query 2
    c.execute(query2)
    #get results to above standard query
    results2 = c.fetchall()
except Exception:
    print("<p><font color=red><b>Error Query 2</b></font></p>")

try:
    #execute query 3
    c.execute(query3)
    #get results to above standard query
    results3 = c.fetchall()
except Exception:
    print("<p><font color=red><b>Error Query 3</b></font></p>")

c.close()
connection.close()


#get all the candidate-faculty pair ids currently in the database which will be used in the section that checks and uses form data
cfids = [cf[0] for cf in results3]

#retrieve form data
form = cgi.FieldStorage()

#if form is empty, then it's possible that everything is to be deleted from the Assignment table
#if not form:
#     if results3:
#          truncateStatement = "DELETE FROM Assignment;"
#          connection = sqlite3.connect('db/BRITEREU.db')
#          c = connection.cursor()
#          c.execute(truncateStatement)
#          connection.commit()

#check what checkboxes are checked
#if checkbox was selected that was not previously selected - insert those pairs into the Assignment table
#if checkbox is no longer selected - delete those pairs from the Assignment table
if form:
     res3 = [pair for pair in cfids]
     pairlist = form.getlist("cf")
     #find pairs that are in the selected list (pairlist) and not in the current database list (res3)
     tobe_inserted = list(set(pairlist) - set(res3))
     tobe_inserted = [tuple(i.split("_")) for i in tobe_inserted]
     #find pairs that are not in the selected list(pairlist) and are in the current database list (res3)
     tobe_removed = list(set(res3) - set(pairlist))
     tobe_removed = [tuple(map(int, i.split("_"))) for i in tobe_removed]
     if tobe_inserted or tobe_removed:
         connection = sqlite3.connect('db/BRITEREU.db')
         c = connection.cursor()
         for pair in tobe_inserted:
             insertStatement = "INSERT INTO Assignment(cid, pid) VALUES (%s, %s);" % pair
             c.execute(insertStatement)
             connection.commit()
         for pair in tobe_removed:  
             deleteStatement = 'DELETE FROM Assignment WHERE cid ="%s" and pid ="%s";' % pair
             c.execute(deleteStatement)
             connection.commit()
         c.close()
         connection.close()


#query the database again to now get all updated pairs
query4 = 'SELECT cid || "_" || pid, assigned_at FROM Assignment ORDER BY(cid);'

connection = sqlite3.connect('db/BRITEREU.db')
c = connection.cursor()
try:
    #execute query 1 
    c.execute(query4)
    #get results to above standard query
    results4 = c.fetchall()
except Exception:
    print("<p><font color=red><b>Error Query 4</b></font></p>")


#form action for user to submit checkboxes selections 
print('''<form name="form1" id="form1" action="https://bioed.bu.edu/cgi-bin/students_21/group_proj/group_K/assign_candidate.py" method="post" >''')
print('<table id=Candidate class="dataframe">')
print("<tr><th>Candidate ID</th><th>Candidate Name</th>")

#gets list of faculty
#adds all the faculty who are in the database as columns
for faculty in results2:
    print("<th>%s</th>") % faculty[2]
print("</tr>")


#get the Project IDs for the projects so that you concatenate to the CID to formulate a value pair
pids = [faculty[0] for faculty in results2]

#added proper URL for reference to reviewer page
#print the candidate table with a checkbox for each faculty member
for row in results1:
    print('''<tr><td><a href="https://bioed.bu.edu/cgi-bin/students_21/group_proj/group_K/reviewer.py?AID=%s">%s</a></td><td>%s %s</td>''') % (row[0], row[0], row[1], row[2]) 
    for f in pids:
        for cf_pair in results4:
            if (str(row[0])+"_"+str(f)) in cf_pair:
                print('<td><input title="%s GMT" type="checkbox" name="cf" value=%s checked="checked" />rank</td>') %  (cf_pair[1], (str(row[0])+"_"+str(f)))
                break
        else:
            print('<td><input type="checkbox" name="cf" value=%s /></td>') %  (str(row[0])+"_"+str(f))
    print("</tr>")


#add submit button for assigning faculty to candidates
print('<input type="submit" value="Assign Candidates" /><br /><br />')

#end form
print("</form>")



#filtering section for the table
print("</table>")
print('''<script src="https://bioed.bu.edu/students_21/group_proj/group_K/tablefilter/tablefilter.js"></script>''')
print('''<script data-config="">
 var filtersConfig = {
     base_path: 'https://bioed.bu.edu/students_21/divyas3/tablefilter/',
  auto_filter: {
                    delay: 110 //milliseconds
              },
              filters_row_index: 1,
              state: true,
              alternate_rows: true,
              rows_counter: true,
              btn_reset: true,
              status_bar: true,
              msg_filter: 'Filtering...'
            };
            var tf = new TableFilter(Candidate, filtersConfig);
            tf.init();
          </script>''')

print("</body> </html>")
