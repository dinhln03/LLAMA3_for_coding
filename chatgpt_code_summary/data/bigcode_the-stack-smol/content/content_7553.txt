#----------------#
# Name: Mod_Obj  #
# Author: Photonic   #
# Date:U/N 	   #
#----------------#

import urllib2
import os
import sys

# Class Mod defines the mod object
'''
#############################
# Class for "mod" objects.  #
# Used to store all data    #
# related to mods.          #
#############################
'''
class Mod:

	def __init__(self, name, url):
		self.name 		= name						# Stores the name of the mod (Supplyed by the suer)
		self.filename 	= name + '.jar'				#  Need to make this a part of the entire file. But I'll do it tomorrow.
		self.url 		= url						# Stores the URL of the mod (supplyd by the user)
		self.state 		= 0							# Used to indecate if the mod wass downloaded succssfully
		self.installed 	= 0
	#Defines the alternative constructor "create_mod"
	'''
	#####################################################################################################
	# create_mod takes in a string that has two values separated by semicolons i.e ("name:url")         #
	# and splits them into the vars, name and url which is then fed into the Class "Mod"                #
	# to create an instance of it's self. i.e "mod = Mod.mod_create("IC2:ic2.download.net")"            #
	#####################################################################################################
	'''
	@classmethod
	def create_mod(cls, string):
		name, url = string.split(':')								# Grab the string var and split it at ":" then store the resole inside "name" and "url"
		return cls(name, 'https://' + url)	 						# Create instance of the class "Mod" and returns it.

	# Defines the fetch function "download_mod"
	'''
	#########################################################################################
	#download_mod takes 'name' and, 'url' then changes the directory to 'temp'              #
	#then creates a Request header using urllib2 and saves it inside 'request'              #
	#then it adds a user agent using "add_header". It opens the url and saves it inside		#
	#response then gets data from responde till there is no data left.						#
	#########################################################################################
	'''
	def download_mod(self):
		chunk_size = 16 * 1024										# How many bytes to read from the buffer at once

		# Attempt to change current directory to 'temp'
		try:
			os.chdir('temp')
		except:														# If can't change to "temp" directory, exit the program
			print("Error: Could not access directory 'temp', not present or, access denied.\n")
			sys.exit()

		request = urllib2.Request(self.url)									# Create a "Request" and save inside of the var 'raquest'
		request.add_header('User-agent', 'Mozilla 5.10')					# add a user agent to the header to avoid the pesky bureaucrats attempts at stopping are bot. hehehehe

		#Attempt to Open the URL
		try:
			print("Attempting to connect to ---> {}".format(self.url))
			response = urllib2.urlopen(request)								# Attempt to open URL@'url' and saves it in 'response'
			print("Connected to ---> {}\n".format(self.url))
		except:
			print("Error: Could not connect to ---> {}\nConnection Dropped, blocked or, nothing is at {}\n".format(self.url, self.url))
			sys.exit()

		# This is for debugging
		#resp_info = response.info()		# Get header info
		#print(resp_info)					# <<< For debugging

		# Attempt to download file from URL.
		try:
			print("Opening file to write to.\n")
			with open(self.filename, 'wb') as file_handler:			# Open file as 'file_handler'
				print("Downloading {},\nThis may take a while.".format(self.filename))
				while True:
					chunk = response.read(chunk_size)				# Read 'chunk_size' from buffer into the var 'chunk'
					if not chunk:break								# If no more data from 'chunk' break out of loop
					file_handler.write(chunk)						# Write var 'chunk' to file_handler
			self.state = True										# Return True if mod succssfully downloaded
			print("Succssfully downloaded {}\n".format(self.filename))
			os.chdir('..')
		except:
			os.chdir('..')
			print("Error: Could not download {}@{}\n".format(self.filename, self.url))
			self.state = False 										# Return "False" if a Error was encoted while attempting to download the mod file
																	# Return to sorce dir
	#------------------------------------------------------- #
	# Checks to seee if mod was downloaded succssfully	     #
	# then changes the directory to temp installs mod inside #
	# Inside var 'path' then deletes 'filename' from temp    #
	# Then delete var 'filename' inside "temp".			     #
	#------------------------------------------------------- #

	def install_mod(self, path):
		if self.state:																									# Checks to see if the mod was downloaded succsfully
			try:
				os.chdir("temp")																						# Attempts to change the directory to temp
			except:
				print("Error: Could not access directory 'temp', not present or, access denied.\n")
			try:
				print("Attempting to Install {}.".format(self.filename))
				os.rename("{}\{}".format(os.getcwd(), self.filename), "{}\{}".format(path, self.filename))				# Attempts to copy the file to 'path'
				print ("Mod Succssfully Installed.\n")
				os.chdir('..')
				return 1
			except:
				print("Error: Problem Installing mod, file or directory not found or, insufficient privileges\n")
				os.chdir('..')
				return 0

			try:
				os.chdir('temp')
				os.remove(self.filename)
				os.chdir('..')
			except:
				os.chdir('..')
		else:
			print('Skipping {}\n>>>>\n'.format(self.filename))
