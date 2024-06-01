#created by Angus Clark on 8/01/2017
# toDo incoperate the saving program into this_dir

import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '130.56.253.43'
print host # remove when done debugging
port = 5201 # edit when port for comm is decided

s.bind((host,port))

f = open('temp.json','wb')
s.listen(5)

while True:
    
    c, addr = s.accept()
    while(l):
        f.write(l)
        l = c.recv(1024)
    f.close()
    c.close()