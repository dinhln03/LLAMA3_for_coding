#!/usr/bin/python3

import sys
import json
import getopt
import os
import jsonschema
import subprocess
 
if os.geteuid() != 0:
  print('You must be a root user')
  sys.exit(72)

json_file = ''
nginx_conf = '/etc/nginx/nginx.conf'
schema_file = ''
test = False

#------Parse command-line options------
def usage():
  print ('Usage: ' + sys.argv[0] + ' -j json_file [-c nginx_ conf] [-s schema_file] [-t] [-v] [-h]')
  print ('  options:')
  print ('    -j json_file      : JSON file (required option)')
  print ('    -c nginx_conf     : Nginx config file (default: /etc/nginx/nginx.conf)')
  print ('    -s schema_file    : JSON schema file')
  print ("    -t                : Test Nginx config file by command '/usr/sbin/nginx -t -c <nginx.conf>'")
  print ('    -v                : Version')
  print ('    -h                : Show this help page')

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:], 'hvtj:c:s:')
except getopt.GetoptError as err:
  print(err)
  usage()
  sys.exit(73)
if len(args) != 0:
  print('Incorrect options: ' + ' '.join(args))
  usage()
  sys.exit(74)
else:
  for o, a in opts:
    if o == '-h':
      usage()
      sys.exit()
    elif o == '-v':
      print('version: 0.0.1')
      sys.exit()
    elif o == '-t':
      test = True
    elif o == '-j':
      json_file = a
    elif o == '-c':
      nginx_conf = a
    elif o == '-s':
      schema_file = a

if json_file == '':
  print('JSON file is required')
  usage()
  sys.exit(75)

#------Get json and schema data------
try:
  fh = open(json_file, 'r')
except IOError:
  print("Could not opent the file '{0}' for reading".format(json_file))
  sys.exit(76)
data=json.load(fh)
fh.close()

if schema_file != '':
  try:
    fh = open(schema_file, 'r')
  except IOError:
    print("Could not opent the file '{0}' for reading".format(schema_file))
    sys.exit(77)
  schema=json.load(fh)
  fh.close()
  try:
    jsonschema.validate(data, schema)
  except Exception as e:
    print(e)
    sys.exit(78)

#------Nginx functions------
def pcrejit():
  try:
    output = subprocess.check_output('/usr/sbin/nginx -V', stderr=subprocess.STDOUT, shell=True)
    if output.decode().find('--with-pcre-jit') != -1:
       return 'on'
    else:
       return 'off'
  except Exception:
    return 'off'

def test_conf():
  if test:
    try:
      output = subprocess.check_output('/usr/sbin/nginx -t -c ' + nginx_conf, stderr=subprocess.STDOUT, shell=True)
      print(output.decode())
    except Exception as e:
      print(e)

#------Test 'location /'------
location_root_test = []
for server in data.get('http').get('server'):
  for location in server.get('location'):
    location_root_test.append(location.get('URI'))
if '/' not in location_root_test:
  print("There is not 'location /' in JSON file")
  sys.exit(79)

#------Make Nginx config file------
try:
  fh = open(nginx_conf, 'w')
except IOError:
  print("Could not open the file '{0}' for writing".format(nginx_conf))
  sys.exit(78)

fh.write( 'user ' + json.dumps(data.get('user')) + ';\n' )

fh.write( 'worker_processes ' + json.dumps(data.get('worker_processes')) + ';\n' )

fh.write( 'error_log ' + json.dumps(data.get('error_log').get('file')) + ' '
                       + json.dumps(data.get('error_log').get('level')) + ';\n' )

fh.write( 'pid ' + json.dumps(data.get('pid')) + ';\n' )

fh.write( 'pcre_jit ' + pcrejit() + ';\n' )

fh.write( 'events { worker_connections ' +  json.dumps(data.get('events').get('worker_connections')) + '; }\n' )

fh.write( 'http {\n')
fh.write( '  include ' + json.dumps(data.get('http').get('include')) + ';\n' )
fh.write( '  default_type ' + json.dumps(data.get('http').get('default_type')) + ';\n' )
fh.write( '  log_format ' + json.dumps(data.get('http').get('log_format').get('name')) + " "
                          + json.dumps(data.get('http').get('log_format').get('string')) + ";\n" )
fh.write( '  access_log ' + json.dumps(data.get('http').get('access_log').get('file')) + ' '
                          + json.dumps(data.get('http').get('access_log').get('name')) + ';\n' )

for server in data.get('http').get('server'):
  fh.write('    server {\n')
  fh.write('      listen ' + json.dumps(server.get('listen')) + ';\n')
  fh.write('      server_name ' + json.dumps(server.get('server_name')) + ';\n')

  # noindex 'location = /robots.txt'
  for extra in server.get('extra', []):
    if extra == 'noindex':
      fh.write('      location = /robots.txt {\n')
      fh.write('        default_type "text/plain";\n')
      fh.write('        return 200 "User-agent: *\\nDisallow: /";\n')
      fh.write('      }\n')

  for location in server.get('location'):
    fh.write('      location ' + location.get('modifier') + ' '
                                + location.get('URI') + ' {\n')
    for configuration in location.get('configuration'):
      if configuration == 'proxy_set_header':
        for proxy_set_header in location.get('configuration').get(configuration):
          fh.write('        proxy_set_header ' + proxy_set_header.get('field') + ' '
                                               + json.dumps(proxy_set_header.get('value')) + ';\n')
      elif configuration == 'return':
        fh.write('        return ' + location.get('configuration').get(configuration).get('code') + ' '
                                   + json.dumps(location.get('configuration').get(configuration).get('text')) + ';\n')
      else:
        fh.write('        ' + configuration + ' ' + json.dumps(location.get('configuration').get(configuration)) + ';\n')
    fh.write( '      }\n' )
  fh.write( '    }\n' )

for upstream in data.get('http').get('upstream'):
  fh.write('    upstream ' + json.dumps(upstream.get('name')) + ' {\n')
  for server in upstream.get('server'):
    fh.write('      server ' + json.dumps(server.get('address')))
    for parameter in server.get('parameters'):
      fh.write(' ' + json.dumps(parameter))
    fh.write(';\n')
  fh.write( '    }\n' )

fh.write( '}\n')

fh.close()

test_conf()
