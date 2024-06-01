#!/usr/bin/env python3
import cgi, cgitb, os, storage, shutil, time, sys, atexit
def deltemp():
    os.remove("_/3dsthemes/tmp.zip")
from libs import zip
cgitb.enable()
from libs.session import Session
from libs import smdh
session=Session()
if not session.isLoggedIn():
    raise ValueError("Must be logged in to upload a file")

form = cgi.FieldStorage()
if not "title" in form:
    raise ValueError("Title is missing")
if not "desc" in form:
    raise ValueError("Description is missing")
if not "file" in form:
    raise ValueError("File is missing")
fileitem = form["file"]
if not fileitem.file:
    raise ValueError("No file uploaded?")
#Check if an upload is in progress
if os.path.isfile("_/3dsthemes/tmp.zip"):
    raise ValueError("An upload is in progress. Please wait a little before reuploading.")
atexit.register(deltemp)
#OK, we're onto something
outpath = "_/3dsthemes/tmp.zip"
fout = open(outpath, "wb")
for f in range(21):
    if f == 20:
        fout.close()
        raise ValueError("File too big.")
    chunk = fileitem.file.read(1000000)
    if not chunk: break
    fout.write(chunk)
fout.close()
tid=storage.count("themes")
dirname = "_/3dsthemes/%i/"%(storage.count("themes"))
try:
    os.mkdir(dirname)
except:
    shutil.rmtree(dirname)
    os.mkdir(dirname)
zip.unzip("_/3dsthemes/tmp.zip", dirname)
try:
    os.rename(dirname+"preview.png", dirname+"Preview.png")
except:
    pass
testfile = smdh.SMDH(dirname+"info.smdh")
#Will throw an exception if the file doesn't exist or isn't valid.
#Put theme into database. This is done last to prevent 'ghost themes'
title=cgi.escape(form["title"].value)
markdown=cgi.escape(form["desc"].value)
author=session.getUserName()
date=int(time.time())
aid=tid
storage.append("themes",{"title":title, "markdown":markdown, "author":author, "date":date, "aid":aid}) #Write
sys.stdout.buffer.write(("Content-type: text/html\r\n\r\n<html><head><script>window.location.replace(\"index.py\");</script></head></html>").encode('utf8'))
print("Test?")
