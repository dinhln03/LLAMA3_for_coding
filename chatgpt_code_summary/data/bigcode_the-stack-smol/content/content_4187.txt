import loquis
import subprocess

@loquis.command
def run(query,*args):
	
	try:
		L=[query.lower()]+list(args)
		print(L)
		return [subprocess.check_output(L)]
	except:
		return ["Failed to run command"]

languages={'en':{'run':run}}
