import os
import subprocess


usersList = subprocess.check_output("grep '/home/' /etc/passwd | cut -d: -f1", shell=True)
users =  usersList.splitlines()

totalFails = 3

for user in users:
	fails = subprocess.check_output("cat /var/log/auth.log | grep '" + user.decode('UTF-8') + " ' | grep 'ssh.*Failed' | wc -l", shell=True)
	if(int(fails.decode('UTF-8')) >= totalFails):
		os.system("passwd " + user.decode('UTF-8') + " -l")
