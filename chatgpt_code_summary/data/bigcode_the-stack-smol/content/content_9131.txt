# Connection libraries
import os
import shutil
import re

# Class create project
class Create:
	def __init__(self, path):
		self.path = path

	# Create project
	def createProject(self, name):
		if not os.path.isdir(self.path + name):
			shutil.copytree("launcher/shablon/", self.path + name)
		else:
			n, a = os.listdir(path=self.path), []
			for s in n:
				if s.find("new") != -1: a.append(s)
			shutil.copytree("launcher/shablon/", self.path + name + str(len(a)))

	# Delete project
	def deleteProject(self, name):
		shutil.rmtree(self.path+name)