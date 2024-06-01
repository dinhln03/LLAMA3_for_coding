#!/usr/bin/python
import cgi
import cgitb
import json
import parse_enumeration

cgitb.enable()


form = cgi.FieldStorage() 

# Get data from fields
callback = form.getvalue('callback')
email = form.getvalue('email')

if (email is None):
    email = "<ul><li>hello, world!</li></ul>"

print "Content-type: application/json"
print 
response = parse_enumeration.parse_enumerations(email)
d = json.JSONEncoder().encode((response))
if (callback):
    print callback+'(' + d + ');'
else:
    print d
