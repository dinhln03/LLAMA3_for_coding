#!/usr/bin/python 

import socket
import fcntl
import struct
import os

ip=socket.gethostbyname(socket.gethostname())
hostname=socket.gethostname()

#def get_ip_address(ifname):
#    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#    return socket.inet_ntoa(fcntl.ioctl(
#        s.fileno(),
#        0x8915,  # SIOCGIFADDR
#        struct.pack('256s', ifname[:15])
#    )[20:24])

#ip=get_ip_address('ens192')


file = open("/root/inventory", "w")
file.write("[minicloud]\n")
file.write( hostname + " ansible_ssh_host=" + ip + " ansible_ssh_user=root ansible_ssh_port=22 ansible_ssh_pass=\"passw0rd\"\n")
file.close()

