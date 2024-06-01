#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: David McCue

import sqlite3, re, os
from bottle import route, run, debug, template, request, static_file, error, response

db_filename=os.path.dirname(os.path.realpath(__file__)) + '/db/kickstarter.db'

# Validate currency values
def valid_amount(amount):
  amount = re.sub(r"\$","",amount)
  amount_regex=re.compile('^[0-9]+(\.[0-9]{2})?$')
  if amount_regex.match(amount) and float(amount) > 0.0:
    return True


# Validate names
def valid_name(name):
  name_regex=re.compile('^[A-Za-z0-9-_]{4,20}$')
  if name_regex.match(name):
    return True


# Calculate luhn checksum
# Credit: wikipedia
def luhn_checksum(card_number):
  def digits_of(n):
    return [int(d) for d in str(n)]
  digits = digits_of(card_number)
  odd_digits = digits[-1::-2]
  even_digits = digits[-2::-2]
  checksum = 0
  checksum += sum(odd_digits)
  for d in even_digits:
    checksum += sum(digits_of(d*2))
  return checksum % 10


# Validate credit card
def valid_creditcard(creditcard):
  creditcard_regex=re.compile('^[0-9]{1,19}$')
  if creditcard_regex.match(creditcard):
    if luhn_checksum(creditcard) == 0:
      return True



# retrieve project by name
@route('/project/<project>')
def project_list(project):

  conn = sqlite3.connect(db_filename)
  c = conn.cursor()
  c.execute("SELECT id, name, target FROM 'project' WHERE name = '" + project + "';")
  result = c.fetchall()
  c.close()

  if result:
    c = conn.cursor()
    c.execute("SELECT name, amount FROM 'transaction' WHERE projectid = " + str(result[0][0]) + ";")
    backers = c.fetchall()
    c.close()
    return {'project': result[0], 'backers': backers}

  else:
    response.status = 400
    return {'msg':'ERROR: Project name not found'}


# add project
@route('/project', method='POST')
def project_new():

  project = request.json
  print "DEBUG: " + str(project)

  if not valid_name(project['name']):
    response.status = 400
    return {'msg':'ERROR: Project name validation error'}
  if not valid_amount(project['target']):
    response.status = 400
    return {'msg':'ERROR: Project target validation error'}

  conn = sqlite3.connect(db_filename)
  c = conn.cursor()
  c.execute("SELECT name FROM project WHERE name = '" + project['name'] + "';")
  result = c.fetchall()
  c.close()
    
  if result:
    response.status = 400
    return {'msg':'ERROR: Project name already exists'}
    
  conn = sqlite3.connect(db_filename)
  conn.execute("INSERT INTO 'project' (name, target) VALUES ('" + project['name'] + "', '" + project['target'] + "');")
  conn.commit()
  conn.close()

# add transaction
@route('/back', method='POST')
def back_new():

  back = request.json

  # Validate inputs
  if not valid_name(back['name']):
    response.status = 412
    return {'msg':'ERROR: Backer name validation error'}
  if not valid_amount(back['amount']):
    response.status = 412
    return {'msg':'ERROR: Backer amount validation error'}        
  if not valid_creditcard(back['cc']):
    response.status = 400
    return {'msg':'ERROR: This card is invalid'}

  # Check credit card is unique to this user name
  conn = sqlite3.connect(db_filename)
  c = conn.cursor()
  c.execute("SELECT name FROM 'transaction' WHERE cc = '" + back['cc'] + "' AND name != '" + back['name'] + "';")
  result = c.fetchall()
  c.close()

  if result:
    response.status = 400
    return {'msg':'ERROR: That card has already been added by another user!'}

  # Get project id from name
  c = conn.cursor()
  c.execute("SELECT id FROM 'project' WHERE name = '" + back['projectname'] + "';")
  result = c.fetchall()
  c.close()

  if not result:
    response.status = 400
    return {'msg':'ERROR: Unable to find project name'}
 
  back['projectid'] = result[0][0]
  conn.execute("INSERT INTO 'transaction' (projectid, name, cc, amount) VALUES (" + str(back['projectid']) + ", '" + back['name'] + "', " + str(back['cc']) + ", " + str(back['amount']) + ");")
  conn.commit()


# retrieve backer by name
@route('/backer/<backer>')
def backer_list(backer):
  conn = sqlite3.connect(db_filename)
  c = conn.cursor()
  c.execute("SELECT project.name, 'transaction'.name, 'transaction'.amount FROM 'transaction' JOIN project ON 'transaction'.projectid = project.id WHERE 'transaction'.name = '" + backer + "';")
  result = c.fetchall()
  c.close()

  if result:
    return {'backer': result}
  else:
    response.status = 400
    return {'msg':'ERROR: Backer not found'}
    

@error(403)
def mistake403(code):
  response.status = 403
  return {'msg':'ERROR: There is a mistake in your URL'}

@error(404)
def mistake404(code):
  response.status = 404
  return {'msg':'ERROR: This page does not exist'}


#debug(True)
run(reloader=True, host='0.0.0.0')
