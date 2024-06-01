# a simple script to rename multiple files

import os
import re
path = 'myimages/'
files = os.listdir(path)
files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
for i, file in enumerate(files):
    os.rename(path + file, path + "rename_{}".format(i)+".jpg")
    
print('done!')
