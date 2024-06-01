#!/usr/bin/env python

"""Convert *.json, *.csv and other text data files to js for local use and avoid ajax call.
"""

import optparse
from os import listdir
from os.path import abspath, isfile, isdir, join, splitext, basename
import json;

#curdir = os.path.abspath('.')
curdir = "."
filter_text_ext = [".json", ".csv"]
filter_binary_ext = []

def jsfy_file(path, basedir, fout):
    fname = basename(path)
    if(fname.startswith(".")):
        return
    #print(path, basedir)
    if(not path.startswith(basedir)):
        return

    filename, extname = splitext( path )
#print( extname )
    if(extname in filter_text_ext):
        res_key = path[ len(basedir) : ]
        print( res_key + " -> " + path )
        fin = open(path, "r")
        txt = json.dumps( fin.read() )
        fout.write("jsfy_res[\"" + res_key + "\"] = " + txt + ";\n\n");
#elif(extname in filter_binary_ext):
#
    pass

def jsfy_dir(path, basedir, fout):
    if(not path.endswith("/")):
        path = path + "/"
    fname = basename(path)
    if(fname.startswith(".")):
        return
    #print(path, basedir)
    if(not path.startswith(basedir)):
        return
    #print( path + ":" )
    for f in listdir(path):
        subpath = join(path,f)
        if( isfile(subpath) ):
            jsfy_file(subpath, basedir, fout)
        elif( isdir(subpath) ):
            jsfy_dir(subpath, basedir, fout)

def main():
    """The entry point for this script."""

    usage = """usage: %prog [dir] [-b basedir] [-o jsfile]
        example:
        %prog
        %prog assets -o js/jsfy_res.js
        """

    parser = optparse.OptionParser(usage)
    parser.add_option("-b", "--base", dest="basedir", help="base dir")
    parser.add_option("-o", "--output", dest="outputpath", help="export js file path")

    (options, args) = parser.parse_args()

    if( isinstance(options.basedir, str)):
        basedir = options.basedir
    else:
        basedir = "."

    basedir = abspath(basedir)

    if( isinstance(options.outputpath, str)):
        outputpath = options.outputpath
    else:
        outputpath ="./jsfy_res.js"

    fout = open( outputpath, "w" )
    fout.write("// generated with jsfy.py, v0.1 (https://github.com/floatinghotpot/jsfy)\n\n" )
    fout.write("var jsfy_res = jsfy_res || {};\n\n" )

    if(not basedir.endswith("/")):
        basedir = basedir + "/"

    for f in args:
        f = abspath(f)
        if( isfile(f) ): jsfy_file(f,basedir,fout)
        elif( isdir(f) ): jsfy_dir(f,basedir,fout)

    fout.close()

    # end of main()

if __name__ == "__main__":
    main()

