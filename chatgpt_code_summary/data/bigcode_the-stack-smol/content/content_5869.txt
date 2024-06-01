'''
Given one or more regular expressions on the command line, searches
the PATH for all files that match.

Copyright (C) 2002 GDS Software

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of
the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public
License along with this program; if not, write to the Free
Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
MA  02111-1307  USA

See http://www.gnu.org/licenses/licenses.html for more details.
'''

import sys, getopt, re, os, string

__version__ = "$Id: where.py,v 1.4 2002/08/22 02:25:57 donp Exp $"

ignore_caseG = 0
matches = {}  # They'll get stored in here by filename so that there are
              # no duplicates.
 

def CheckDirectory(dir, regexps):
    '''dir is a directory name, regexps is a list of compiled
    regular expressions.
    '''
    global matches
    currdir = os.getcwd()
    try:
        os.chdir(dir)
        tmp = os.listdir(dir)
        files = []
        for f in tmp:
            if os.path.isfile(f):
                files.append(f)
        for file in files:
            for regexp in regexps:
                if regexp.search(file) != None:
                    matches[dir + "/" + file] = ""
    except:
        sys.stderr.write("Warning:  directory '%s' in PATH not found\n" % dir)
        os.chdir(currdir)

def main():
    global ignore_caseG
    try:
        optlist, regexps = getopt.getopt(sys.argv[1:], "i")
    except getopt.error, str:
        print str
        sys.exit(1)
    for opt in optlist:
        if opt[0] == "-i":
            ignore_caseG = 1
    if len(regexps) == 0:
        print "Usage:  where [-i] regexp1 [regexp2...]"
        print "  regexps are python re style"
        sys.exit(1)
    # Get a list of the directories in the path
    sep = ":"
    key = "PATH"
    if sys.platform == "win32":
        sep = ";"
    if key in os.environ.keys():
        PATH = os.environ[key]
        path = re.split(sep, os.environ[key]) 
    else:
        print "No PATH variable in environment"
        sys.exit(1)
    # Make a list of compiled regular expressions
    regexp_list = []
    for regex in regexps:
        if ignore_caseG:
            regexp_list.append(re.compile(regex, re.I))
        else:
            regexp_list.append(re.compile(regex))
    # Now check each command line regexp in each directory
    for dir in path:
        CheckDirectory(dir, regexp_list)
    list = []
    for key in matches.keys():
        list.append(key)
    list.sort()
    for file in list:
        print string.replace(file, "\\", "/")

main()
