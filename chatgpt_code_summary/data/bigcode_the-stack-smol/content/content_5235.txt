import os
from os.path import splitext

names=os.listdir()
number=1
f_name=""
videos=[]
ext = [".3g2", ".3gp", ".asf", ".asx", ".avi", ".flv", ".m2ts", ".mkv", ".mov", ".mp4", ".mpg", ".mpeg", ".rm", ".swf", ".vob", ".wmv"]

for fileName in names:
    if fileName.endswith(tuple(ext)):
        f_name, f_ext= splitext(fileName)
        videos.append(f_name)

vidIteretor=iter(videos)

for sub in names:
    if sub.endswith(".srt"):
        os.rename(sub, next(vidIteretor)+".srt")
        


