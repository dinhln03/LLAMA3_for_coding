#!/usr/bin/env python3
import sys
import os
import re
import argparse
import requests
from bs4 import BeautifulSoup as bs
version=1.1
print("""\033[1;36m
╦ ╦╔═╗╔╗     ╦═╗╔═╗╔═╗╔╦╗╔═╗╦═╗
║║║║╣ ╠╩╗    ╠╦╝║╣ ╠═╣║║║║╣ ╠╦╝
╚╩╝╚═╝╚═╝────╩╚═╚═╝╩ ╩╩ ╩╚═╝╩╚═
🔗🔥🔗🔥🔗🔥🔗🔥🔗🔥🔗🔥🔗🔥🔗🔥
                                                   --> Coded by FEBIN 🛡️🌐

\033[1;39m""")

def febrev_fuzz(url):
	import requests
	os.system("clear")
	feblist=open("admin-panel.txt","r+")
	text=str(feblist.read())
	adminpages=list(text.split())
	feblist.close()
	print(f"""
[\033[1;37m+\033[1;39m] STARTED CRAWLING TO FIND ADMIN PANEL OF URL : \033[1;34m{url}
	""")
	try:
		if url.startswith("https://") or url.startswith("http://"):
			url=url
		else:
			print("Error : INVALID URL ! URL must start with 'http://' or 'https://'")
			exit()
		
		if url.endswith("/"):
			url=url
			server=requests.get(url).headers.get('Server')
			print(f"\033[1;37m SERVER Type >> {server}")
			print("\n<----------------------------------------------------------------------------------->")
			print(" ")
		else:
			url=f"{url}/"
			server=requests.get(url).headers.get('Server')
			print(f"\033[1;37mSERVER Type >> {server}")
			print("\n<----------------------------------------------------------------------------------->")
			print(" ")
		for i in range(len(adminpages)):
			reqresp=requests.get(f"{url}/{adminpages[i]}",timeout=10)
			if reqresp.status_code == 200:
				print(f"\033[1;39m FOUND  ==> {url}{adminpages[i]} \033[1;34m")
			elif reqresp.status_code == 302:
				print("\033[1;39m FOUND 302  ==> {url}{adminpages[i]} \033[1;34m")
					
			else:
				pass
	except requests.exceptions.ConnectionError:
		print("[\033[1;31m-\033[1;39m] Connection to the Server Failed, May be invalid URL or bad Internet connection. Check Your Internet connection,URL and try again\n ")
	except requests.exceptions.ReadTimeout:
		print("\033[1;31m [\033[1;31m-\033[1;39m] Error : EXECUTION STOPPED DUE TO !TIMED OUT! ERROR, YOUR INTERNET MAY BE DISCONNECTED!!!....EXITTED")
	
	print("\033[1;37m WEB_REAMER Execution Completed. \033[1;33m!HAPPY HACKING! \033[1;34m \n")


def sub_brute(domain,sublist):
	if os.path.isfile(sublist):
		print(f"[\033[1;37m+\033[1;39m] Subdomain wordlist {sublist} loaded -> OK")
		print("")
		pass
	else:
		print(f"[\033[1;31m-\033[1;39m] Wordlist {sublist} not found!!")
		exit()
	sub=open(sublist,"r+")
	subs=sub.read().split("\n")
	sub.close()
	for host in subs:
		try:
			req=requests.get(f"http://{host}.{domain}")
			print(f"\033[1;39m{host}.{domain}     --> \033[1;37m{req.status_code}")
		except requests.exceptions.ConnectionError:
			pass
		except UnicodeError:
			pass
	print("")
	print("[\033[1;37m+\033[1;39m] Finshed!")
	
	print("\033[1;37m WEB_REAMER Execution Completed. \033[1;33m!HAPPY HACKING! \033[1;34m \n")
def wordlistgen(url,filepath):
	import requests
	from bs4 import BeautifulSoup
	print("")
	try:
		webpage=requests.get(url)
		pagedata=webpage.text
		soup=BeautifulSoup(pagedata,"html.parser")
	except requests.exceptions.ConnectionError:
		print("\033[1;31m[-] ERROR CONNECTING THE SERVER...")
		exit()
	for script in soup(["script","style"]):
		script.extract()
	text1=soup.get_text()
	text=str(text1.strip())
	feb=text.split()
	iscount=feb.count('is')
	wascount=feb.count('was')
	arecount=feb.count('are')
	forcount=feb.count('for')
	thecount=feb.count('the')
	ofcount=feb.count('of')
	tocount=feb.count('to')
	try:
		isinit=0
		while isinit<=iscount:
			feb.remove('is')
			isinit=isinit+1
		wasinit=0
		while wasinit<=wascount:
			feb.remove('was')
			wasinit=wasinit+1
		areinit=0
		while areinit<=arecount:
			feb.remove('are')
			areinit=areinit+1
		forinit=0
		while forinit<=forcount:
			feb.remove('for')
			forinit=forinit+1
		theinit=0
		while theinit<=thecount:
			feb.remove('the')
			theinit=theinit+1
		ofinit=0
		while ofinit<=ofcount:
			feb.remove('of')
			ofinit=ofinit+1
		toinit=0
		while toinit<=tocount:
			feb.remove('to')
			toinit=toinit+1
	except ValueError:
		pass
	feb.sort()
	for string in feb:
		count=feb.count(string)
		strinit=0
		while strinit < count:
			feb.remove(string)
			strinit=strinit+1
	feb.sort()
	for i in range(len(feb)):
		try:
			file=open(filepath,"a+")
			file.write("\n"+feb[i])
			file.close()
		except FileNotFoundError:
			homedir=os.environ.get('HOME')
			file=open(f"{homedir}/fr-wordlist.txt","a+")
			file.write("\n"+feb[i])
			file.close()
	if os.path.isfile(filepath):
		print("")
		print(f"\033[1;39m[\033[1;37m+\033[1;39m]Wordlist {filepath} successfully witten")
	else:
		print("\033[1;31m[-]Sorry:Path not Found!! The Path You Specified Doesn't Exist")
		print("So Saved the wordlist as fr-wordlist.txt in the HOME Directory of the current User.....")
	print("\033[1;37m WEB_REAMER Execution Completed. \033[1;33m!HAPPY HACKING! \033[1;34m \n")



def word_analyze(url):
	import requests
	from bs4 import BeautifulSoup
	print("")
	try:
		webpage=requests.get(url)
		pagedata=webpage.text
		soup=BeautifulSoup(pagedata,"html.parser")
	except requests.exceptions.ConnectionError:
		print("\033[1;31m[\033[1;31m-\033[1;39m] ERROR CONNECTING THE SERVER...")
		exit()
	for script in soup(["script","style"]):
		script.extract()
	text1=soup.get_text()
	text=str(text1.strip())
	feb=text.split()
	iscount=feb.count('is')
	wascount=feb.count('was')
	arecount=feb.count('are')
	forcount=feb.count('for')
	thecount=feb.count('the')
	ofcount=feb.count('of')
	tocount=feb.count('to')
	try:
		isinit=0
		while isinit<=iscount:
			feb.remove('is')
			isinit=isinit+1
		wasinit=0
		while wasinit<=wascount:
			feb.remove('was')
			wasinit=wasinit+1
		areinit=0
		while areinit<=arecount:
			feb.remove('are')
			areinit=areinit+1
		forinit=0
		while forinit<=forcount:
			feb.remove('for')
			forinit=forinit+1
		theinit=0
		while theinit<=thecount:
			feb.remove('the')
			theinit=theinit+1
		ofinit=0
		while ofinit<=ofcount:
			feb.remove('of')
			ofinit=ofinit+1
		toinit=0
		while toinit<=tocount:
			feb.remove('to')
			toinit=toinit+1
	except ValueError:
		pass
	feb.sort()
	print("\033[1;32m-"*74)
	print("\033[1;32m|           Words    |     count/frequency    |        Graph              |  ")
	print("\033[1;32m-"*74)
	for string in feb:
		count=feb.count(string)
		for i in range(count):
			feb.remove(string)
		print(f"\033[1;34m| {string + ' ' * (22 - len(string)) + '| '}{str(count) +' ' * (22 - len(str(count)))}|    \033[1;32m{'█' * count}  " )
		print("\033[1;33m-"*74)



def endpoint_harvest(url):
	print(f"[\033[1;37m+\033[1;39m] Collecting Endpoints / Links from the webpage {url}")
	from bs4 import BeautifulSoup
	print("")
	try:
		webpage=requests.get(url)
		pagedata=webpage.text
		soup=BeautifulSoup(pagedata,"html.parser")
	except requests.exceptions.ConnectionError:
		print("\033[1;31m[\033[1;31m-\033[1;39m] ERROR CONNECTING THE SERVER...")
		exit()
		
	endpoint_pattern1=re.compile('(?:href=")(.*?)"')
	endpoint_pattern2=re.compile('(?:src=")(.*?)"')
	endpoint1=endpoint_pattern1.findall(pagedata)
	endpoint2=endpoint_pattern2.findall(pagedata)
	for link in endpoint1:
		print(link.replace("href=","").replace("'","").replace(">","").replace('"','').replace("</"," "))
		
	for src in endpoint2:
		print(src.replace("src=","").replace("'","").replace(">","").replace('"','').replace("</"," "))
		
	print("")
	print("[\033[1;37m+\033[1;39m] Finished!")
		
def param(url):
	from bs4 import BeautifulSoup
	print("")
	try:
		webpage=requests.get(url)
		pagedata=webpage.text
		soup=BeautifulSoup(pagedata,"html.parser")
	except requests.exceptions.ConnectionError:
		print("\033[1;31m[\033[1;31m-\033[1;39m] ERROR CONNECTING THE SERVER...")
		exit()
	params=soup.find_all("input")
	print("[\033[1;37m+\033[1;39m] Extracting Parameters from the WebPage!\n")
	for param in params:
		print(param.get("name"))	
	print("[\033[1;37m+\033[1;39m] Finished!")

parser = argparse.ArgumentParser(description='Parse the domain, wordlist etc..')
parser.add_argument('-link',dest='link', action='store_true',help='Extract Endpoints from url!')
parser.add_argument('-admin',dest='admin', action='store_true',help='Find Admin Panel of the given URL !')
parser.add_argument('-sub',dest='sub', action='store_true',help='Subdomain brute force of the given domain !')
parser.add_argument('-param',dest='param', action='store_true',help='Find hidden parameters from the given URL !')
parser.add_argument('-wordlist',dest='wordlist', action='store_true',help='Create targeted wordlist from the given URL !')
parser.add_argument('-analyze',dest='analyze', action='store_true',help='Analyze words and their frequencies from the given URL !')
parser.add_argument('-u',"--url",dest='url', action='store',help='The URL of the webpage!')
parser.add_argument('-d',"--domain",dest='domain', action='store',help='The domain name for sub domain brute-force!')
parser.add_argument('-w',"--wordlist",dest='list', action='store',help='Extract Endpoints from url!')
parser.add_argument('-o',"--outfile",dest='outfile', action='store',help='Output file to save the generated wordlist!!')
parser.add_argument('-v',"--version",dest='version', action='store_true',help='Version / Update Check !')
args=parser.parse_args()


try:
	if args.link and args.url:
		if args.url.startswith("http://") or args.url.startswith("https://"):
			endpoint_harvest(args.url)
		else:
			print("[\033[1;31m-\033[1;39m] Invalid URL !")
			exit()
	elif args.admin and args.url:
		if args.url.startswith("http://") or args.url.startswith("https://"):
			febrev_fuzz(args.url)
		else:
			print("[\033[1;31m-\033[1;39m] Invalid URL !")
			exit()
	elif args.sub and args.domain and args.list:
		if args.domain.startswith("http://") or args.domain.startswith("https://"):
			print("[\033[1;31m-\033[1;39m] Expected Domain name not URL!")
			exit()
		else:
			sub_brute(args.domain,args.list)
	elif args.wordlist and args.url and args.outfile:
		if args.url.startswith("http://") or args.url.startswith("https://"):
			wordlistgen(args.url,args.outfile)
		else:
			print("[\033[1;31m-\033[1;39m] Invalid URL !")
			exit()
	elif args.analyze and args.url:
		if args.url.startswith("http://") or args.url.startswith("https://"):
			word_analyze(args.url)
		else:
			print("[\033[1;31m-\033[1;39m] Invalid URL !")
			exit()
	elif args.param and args.url:
		if args.url.startswith("http://") or args.url.startswith("https://"):
			param(args.url)
		else:
			print("[\033[1;31m-\033[1;39m] Invalid URL !")
			exit()
	elif args.version:
		print(f"CURRENT VERSION : {version}")
		try:
			verq=requests.get("http://raw.githubusercontent.com/febinrev/web_reamer/master/version")
			ver=float(verq.text.split()[0])
			if ver > version:
				print(f"[\033[1;37m+\033[1;39m] New Version {ver} of WEB_REAMER is available : https://github.com/febinrev/web_reamer.git")
			else:
				print("[\033[1;37m+\033[1;39m] WEB_REAMER is up-to-date!")
		except requests.exceptions.ConnectionError:
			print("[\033[1;31m-\033[1;39m] Error Connecting github !")
	else:
		print("""\033[1;33m
Usage:
\033[1;32m1. Endpoint / Link Extraction:
\033[1;39m ./web_reamer.py -link -u http://sample.com/ \033[1;32m
 
2. Admin Panel fuzzing:
\033[1;39m ./web_reamer.py -admin -u http://sample.com/ \033[1;32m
 
3. Subdomain Brute Force:
\033[1;39m ./web_reamer.py -sub -d sample.com -w subdomains.txt \033[1;32m
 
4. Find hidden parameters from webpage:
\033[1;39m ./web_reamer.py -param -u http://sample.com/ \033[1;32m
 
5. Create Targetted Wordlist from webpage:
\033[1;39m ./web_reamer.py -wordlist -u http://sample.com/ -o outfile_wordlist.txt \033[1;32m
 
6. Analyze Word frequencies from the WebPage :
\033[1;39m  ./web_reamer.py -analyze -u http://sample.com/ \033[1;32m

7. Help :
\033[1;39m  ./web_reamer.py -h \033[1;32m
\033[1;39m  ./web_reamer.py --help \033[1;32m

8. Version / Update Check :
\033[1;39m  ./web_reamer.py -v \033[1;32m
\033[1;39m  ./web_reamer.py --version \033[1;32m
		""")
except KeyboardInterrupt:
	print("\n\033[1;39m[\033[1;31m-\033[1;39m] User Interruption! Exit!")
	exit()





