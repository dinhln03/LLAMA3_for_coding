import os
from subprocess import call

if os.path.isdir("bin/test"):
    call(["fusermount", "-u", "bin/test"])
    os.rmdir("bin/test")
    
os.mkdir("bin/test")
call(["bin/simple", "-f", "bin/test"])
