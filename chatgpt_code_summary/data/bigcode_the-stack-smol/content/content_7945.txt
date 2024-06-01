#!/usr/bin/env python

from netmiko import ConnectHandler
from getpass import getpass
from datetime import datetime

device = {
    'device_type': 'arista_eos',
    'ip': 'arista1.lasthop.io',
    'username': 'pyclass',
    'password': getpass(),
    'global_delay_factor': 5,
    'session_log': 'arista.txt',
} 

start = datetime.now()
net_connect = ConnectHandler(**device)
output = net_connect.send_command('show ip arp')
print(output)

print()
print("Elapsed time: {}".format(datetime.now() - start))
