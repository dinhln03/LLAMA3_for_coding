try:
  from os import system
  from os.path import isdir,isfile
  from time import sleep
  from npc import NPC
  from tutorial import text
  import pack
  import sys
  from requests import get

  if not isfile('./config'):
    open('config','w').write('firmware: https://raw.githubusercontent.com/miko1112/comp9/main/firmware.c9s')

  config={l.split(': ')[0]:l.split(': ',1)[1] for l in open('config').read().split('\n')}

  def connected():
    try:
      get('https://example.com',2)
      return True
    except:
      return False

  hasinternet=connected()

  cls=lambda: system('cls')
  elseflag=False

  variables={}

  help=''' help     Display this screen
   tut      Read tutorial
   cls      Clear screen
   echo     Display text in the console
   exec     Execute a script file
   bind     Bind a set of commands to one command
   var      Set a variable
   list     Shows all active variables
   wait     Stop for a given amount of seconds
   input    Take input into variable
   run      Execute a command
   while    Repeat some code while condition is not zero
   if       Run some code if condition is not zero
   else     Run some code if previous if statement failed
   program  Run an installed program
   install  Installs a package from the internet
   exit     Shut down COMP-9

   Append ~h to a command (<command> ~h) to get some in
   detail help on said command.'''
  def setvar(*a):
    global variables
    variables[a[0]]=' '.join(a[1:])
  def readtutorial(*a):
    page=text[int(a[0])-1]
    cls()
    print(page)
  def loadscript(*a):
    sn=a[0]
    code=open(sn).read()
    execscript(code)
  def binding(*a):
    commands[a[0]]=('User defined command "'+a[0]+'"','seq',' '.join(a[1:]))
  def wait(*a):
    sleep(float(a[0]))
  def takein(*a):
    global variables
    variables[a[0]]=input(' '.join(a[1:])+' ')
  def run(*a):
    if a[0]=='exit':
      cls()
      sys.exit()
    execscript(' '.join(a))
  def whileloop(*a):
    og_condition=a[0]
    while True:
      calcondition=og_condition
      for vn in variables.keys():
        calcondition=calcondition.replace('$'+vn,variables[vn])
      if calcondition!='0':
        execscript(' '.join(a[1:]))
      else:
        break
  def ifstate(*a):
    global elseflag
    if a[0]!='0':
      execscript(' '.join(a[1:]))
      elseflag=False
    else:
      elseflag=True
  def elsestate(*a):
    global elseflag
    if elseflag:
      execscript(' '.join(a))
      elseflag=False
  def program(*a):
    ogscript=open('./packages/'+' '.join(a)+'/main.c9s').read()
    execscript(ogscript.replace('exec .','exec ./packages/'+' '.join(a)))
  def install(*a):
    if pack.installpackage(get(' '.join(a)).content.decode('utf8')):
      print(' Failed to install package')
    else:
      print(' Successfully installed package at '+' '.join(a))
  def uninstall(*a):
    pack.uninstallpackage(' '.join(a))
  def listvar(*_):
    for k in variables.keys():
      print(' '+k)

  commands={
    'help':('Displays all the commands and what they do','disp',help),
    'exit':('Exits COMP-9','func',cls,sys.exit),
    'cls':('Clears the screen','func',cls),
    'echo':('Displays text\n  echo <text>','comp',print),
    'exec':('Executes a script\n  exec <script location>','comp',loadscript),
    'bind':('Binds a sequence of commands to one command\n  bind <name> <command1;command2;...>','comp',binding),
    'tut':('Shows a page of the tutorial\n  tut <page>','comp',readtutorial),
    'var':('Sets a variable\n  var <name> <value>','comp',setvar),
    'wait':('Waits a given amount of seconds\n  wait <time>','comp',wait),
    'input':('Takes input and puts it into a variable\n  input <variable name> <query>','comp',takein),
    'run':('Executes one command\n  run <command>','comp',run),
    'while':('While the given condition is not 0, execute a command\n  while <condition> <command>','comp',whileloop),
    'if':('If the given condition is not 0, execute a command\n  if <condition> <command>','comp',ifstate),
    'else':('If a previous condition proves false, execute a command\n  else <command>','comp',elsestate),
    'program':('Runs an installed program\n  program <name>','comp',program),
    'install':('Installs a package from the internet\n  install <link to raw text>','comp',install),
    'uninstall':('Uninstalls a package\n  uninstall <name>','comp',uninstall),
    'list':('Lists all active variables','comp',listvar),
  }

  def execscript(t):
    for l in t.split('\n'):
      if l.strip()==''or l.strip().startswith('//'):continue
      if l.strip().split(' ')[0]in commands.keys():
        if execomm(commands[l.strip().split(' ')[0]],*l.strip().split(' ')[1:]):
          print(' Bad syntax "'+l+'"')
          break
      else:
        print(' Unknown command "'+l+'"')
        break

  def execomm(c,*i):
    ct=c[1]
    if len(i)>0:
      if i[0]=='~h':
        print(' '+c[0])
        return False
    if ct=='disp':
      print(c[2])
      return False
    elif ct=='func':
      for f in c[2:]:
        f()
      return False
    elif ct=='comp':
      try:
        proparg=list(i)
        for ind in range(len(proparg)):
          for vn in variables.keys():
            proparg[ind]=proparg[ind].replace('$'+vn,variables[vn])
        c[2](*proparg)
        return False
      except SystemExit:
        sys.exit()
      except Exception as e:
        return True
    elif ct=='seq':
      execscript(c[2].replace(';','\n'))
      return False

  system('title COMP-9')
  execscript('install '+config['firmware']+'\nprogram __SYSTEM')
  #execscript('exec ./system_scripts/system.c9s')
  #cls()
except KeyboardInterrupt:pass
