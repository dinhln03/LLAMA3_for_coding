#!/usr/bin/python

import gdbm
import sys
import os

db_filename = "aclhistory.db"
example_filename = "HOMEOFFICEROLL3_20180521.CSV"
example_status = "D"

if len(sys.argv) != 3:
	scriptname = os.path.basename(str(sys.argv[0]))
	print "usage:", scriptname, "<FILENAME>", "<STATUS>"
	print "\t Pass in the filename and status to be set in the .db file(" + db_filename + ")"
	print "\t Example: ", scriptname, example_filename, example_status
	print "\t to set file", example_filename, "=", example_status, "in", db_filename
	os._exit(1)

file_to_set = str(sys.argv[1])
status_to_set = str(sys.argv[2])

db_file = gdbm.open(db_filename,'c')

for f in db_file.keys():
	if f == file_to_set:
		print "Updating the key", f
		db_file[f] = status_to_set
	print "File", f, "State", db_file[f]
