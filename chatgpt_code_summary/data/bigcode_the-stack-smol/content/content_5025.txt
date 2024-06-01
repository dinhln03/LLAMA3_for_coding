from importlib.resources import path
import sys
import os
import shutil
from git import Repo
from subprocess import call
from git import RemoteProgress
import git
from tqdm import tqdm
from pathlib import Path


dir_path = (os.path.expanduser('~/Documents') + "\server")
os.chdir(dir_path)
gitaddress = str("https://github.com/0xol/server")
print("what server version would you like to install")
print("format is 'client-version'")
print("example 'forge-1.16.5' or 'vanilla-1.7.10'")
print("for lists of supported server version check https://github.com/0xol/server and check under branches")
branch = input()

os.system("del /F /S /Q /A .git")
os.system("del /F /S /Q /A .git") #just in case the program didnt kill it the first time

folder = dir_path
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))




class CloneProgress(RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def update(self, op_code, cur_count, max_count=None, message=''):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()


print(dir_path)
Repo.clone_from(gitaddress, dir_path , branch=branch, progress=CloneProgress())

