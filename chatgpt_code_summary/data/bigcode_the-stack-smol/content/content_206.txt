#!/usr/bin/env python3
import sys
import os
import argparse
scriptpath = os.path.abspath(os.path.dirname(__file__))
includepath = os.path.dirname(scriptpath)
sys.path.insert(0, includepath)
from audio.audiofilefactory import AudioFileFactory
from audio.audioconversionservice import AudioConversionService
from filesystem.filelist import FileList

parser = argparse.ArgumentParser(description="Convert music files", epilog="File types are auto-derived from the filename extensions.")
parser.add_argument("source_path", help="The source path")
parser.add_argument("destination_path", help="The destination path")
parser.add_argument("list_of_favourites", help="The list of favourites")
args = parser.parse_args()
source_path = args.source_path
destination_path = args.destination_path
list_of_favourites = args.list_of_favourites
with open(list_of_favourites) as f:
    content = f.readlines()
content = [x.strip() for x in content]

factory = AudioFileFactory()
for favourite in content:
  statvfs = os.statvfs(destination_path)
  free_space = statvfs.f_bavail * statvfs.f_bsize
  print("Space left: " + str(free_space / 1024 / 1024 / 1024) + " Gb")
  if free_space < 700 * 1024 * 1024:
    print("Skipping " + favourite + ", less than 700 Mb left on device (" + str(free_space / 1024 / 1024) + " Mb)")
    continue 
  target_dir = os.path.join(destination_path, favourite)
  if os.path.isdir(target_dir):
    print("Skipping " + favourite + ", path already exists")
    continue
  os.mkdir(target_dir)
  list = FileList(None, factory)
  list.add_path_to_list(os.path.join(source_path, favourite))
  for f in list:
    source_file_path = f.get_path()
    destination_file_path = os.path.join(target_dir, os.path.splitext(os.path.basename(source_file_path))[0] + ".wav")
    destination_file = factory.create_file(destination_file_path)
    AudioConversionService().convert_audio_file(f, destination_file)
