import os

imagedir = "/Users/titlis/cogsci/projects/stanford/projects/overinformativeness/experiments/5_norming_object_typicality/images"

for t in os.listdir(imagedir):
	if not t.startswith("."):
		for i in os.listdir(imagedir+"/"+t):
			if not i.startswith("."):
				print "{"
				print "\"item\": \""+i[0:-4]+"\","
				print "\"objecttype\": \""+t+"\""
				print "},"