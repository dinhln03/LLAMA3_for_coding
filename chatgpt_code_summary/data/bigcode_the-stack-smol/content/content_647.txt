import sys, imp, atexit, os
sys.path.append("/home/courses/cs3214/software/pexpect-dpty/");
import pexpect, shellio, signal, time, os, re, proc_check

# Determine the path this file is in
thisdir = os.path.dirname(os.path.realpath(__file__))


#Ensure the shell process is terminated
def force_shell_termination(shell_process):
	c.close(force=True)

# pulling in the regular expression and other definitions
# this should be the eshoutput.py file of the hosting shell, see usage above
definitions_scriptname = sys.argv[1]
def_module = imp.load_source('', definitions_scriptname)

# you can define logfile=open("log.txt", "w") in your eshoutput.py if you want logging!
logfile = None
if hasattr(def_module, 'logfile'):
    logfile = def_module.logfile

#spawn an instance of the shell, note the -p flags
c = pexpect.spawn(def_module.shell,  drainpty=True, logfile=logfile, args=['-p', thisdir])

atexit.register(force_shell_termination, shell_process=c)


# set timeout for all following 'expect*' calls to 5 seconds
c.timeout = 5 

#############################################################################
#
# Actual Test
assert c.expect(def_module.prompt) == 0, "Shell did not print expected prompt (1)"

c.sendline("systemInfo")

assert c.expect('------------------------------------------------\r\n') == 0, "Shell did not print out expected values";
assert c.expect(def_module.prompt) == 0, "Shell did not print expected prompt (2)"
shellio.success()
