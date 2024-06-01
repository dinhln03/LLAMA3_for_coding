"""
This module provides an interface for reading and writing to a HackPSU RaspberryPi Scanner config file

Methods:
	getProperty(configFile, prop)
		Get a property from a config file by reading the config file until the desired property is found
	setProperty(configFile, prop, value)
		Set a property by updating the config file (requries a total rewrite of the config file)
		
	getProperties(configFile)
		Read all properties into a dictionary, which is returned to the user
	setProperties(configFile, dict)
		Overwrite the configFile with a new configFile generated from the dictionary provided

"""

def getProperties(configFile):
	"""
	dictionary getProperties(str)
	
	This funciton reads the entire config file and builds a dictionary from the config file
	
	Args:
		configFile: The configuration file to read from
		
	Returns:
		dictionary: A list of key value pairs from the config file
	
	"""
	dict = {}
	#For each line in the file
	with open(configFile) as file:
		for line in file:
			#Remove leading and trailing whitespace
			line = line.strip()
			#If the line is a comment, skip
			if line.startswith('#'):
				continue
			#Find the equals sign, if not present, skip the line
			loc = line.find('=')
			if loc == -1:
				continue
			#parse out the key and value
			key = line[:loc]
			value = line[loc+1:]
			dict[key] = value
	return dict
	
def setProperties(configFile, dict):
	"""
	void setProperties (str, dictionary)
	
	This function iterates over the entire dictionary and saves each dictionary entry to the specified config file
	
	Args:
		configFile: The file to overwrite with the new configuration
		dict: The dictionary to write
	"""
	#Overwrite the file
	#Foreach key in dictionary write a new line
	with open(configFile, 'w') as file:
		for key in dict:
			file.write(key + '=' + dict[key] + '\n')

def getProperty(configFile, prop):
	"""
	str getProperty(str, str)
	
	This function searches a configFile for a specific property and returns its value
	
	Args:
		configFile: The configuration file to open
		prop: The property to search for
		
	Returns:
		string: The property value if found or None for no value found
	
	"""
	retVal = None
	#Foreach line in the file
	with open(configFile) as file:
		for line in file:
			#Remove leading and trailing whitespace
			line = line.strip()
			#Ignore comment lines
			if line.startswith('#'):
				continue
			#If the line is the desired property, parse and return
			if line.startswith(prop):
				retVal = line.replace(prop, '')
				retVal = retVal.strip()
				retVal = retVal[1:]
				retVal = retVal.lstrip()
				break
	return retVal

def setProperty(configFile, prop, value):
	"""
	void setProperty(str, str, str)
	
	This function searches a config file for the specified propery and updates its value if found.
	If the specified property is not found, then a new line for the property will be created
	
	Args:
		configFile: The configuration file to open and update
		prop: The property key to update 
		value: The new value for the property
	
	"""
	written = False
	with open(configFile) as inFile:
		#Create a temp file to copy into
		tmpHandle, outPath = mkstemp()
		with fdopen(tmpHandle, 'w') as outFile:
			#Foreach line in the original file 
			for line in inFile:
				#If it's the prop line, rewrite the prop line
				if line.startswith(prop):
					outFile.write(prop + '=' + value + '\n')
					written = True
				#Otherwise keep the line as is
				else:
					outFile.write(line)
			#If no update was performed, then add a new line for the prop
			if not written:
				outFile.write(prop + ':' + value + '\n')
	#Move from tmp to actual file
	remove(configFile)
	move(outPath, configFile)
		
			
		
