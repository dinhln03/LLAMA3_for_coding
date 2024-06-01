#!usr/bin/env python
import pyeapi
import yaml
from getpass import getpass
from pprint import pprint
from jinja2 import Template

## Loading the yaml file
with open("arista_connect1.yml") as f:
    device_dict = yaml.load(f)
new_list = []
### Keys in the dictionary stored in a list
for k in device_dict.keys():
    new_list.append(k)
### data and connect for 4 arista switches
intf_vars = {}
connect_dict = {}

arista_1 = device_dict[new_list[0]]
arista_2 = device_dict[new_list[1]]
arista_3 = device_dict[new_list[2]]
arista_4 = device_dict[new_list[3]]
for k,v in arista_1.items():
    if k == 'data':
        intf_vars = arista_1[k]
    else:
        connect_dict[k] = arista_1[k]
connection = pyeapi.client.connect(**connect_dict,password=getpass())
device = pyeapi.client.Node(connection)
interface_config = '''
interface {{ intf_name }}
  ip address {{ intf_ip }}/{{ intf_mask }}
'''
j2_template = Template(interface_config)
output = j2_template.render(**intf_vars)
config = (output.strip('/n')).split('\n')
cfg = config[1:3]
out = device.config(cfg)
print(out)
show_ip_int = device.enable("show ip interface brief")
pprint(show_ip_int)
#### For arista switch 2
for k,v in arista_2.items():
    if k == 'data':
        intf_vars = arista_2[k]
    else:
        connect_dict[k] = arista_2[k]
connection = pyeapi.client.connect(**connect_dict,password=getpass())
device = pyeapi.client.Node(connection)
output = j2_template.render(**intf_vars)
config = (output.strip('/n')).split('\n')
cfg = config[1:3]
out = device.config(cfg)
print(out)
show_ip_int = device.enable("show ip interface brief")
pprint(show_ip_int)
### Arista switch 3
for k,v in arista_3.items():
    if k == 'data':
        intf_vars = arista_3[k]
    else:
        connect_dict[k] = arista_3[k]
connection = pyeapi.client.connect(**connect_dict,password=getpass())
device = pyeapi.client.Node(connection)
output = j2_template.render(**intf_vars)
config = (output.strip('/n')).split('\n')
cfg = config[1:3]
out = device.config(cfg)
print(out)
show_ip_int = device.enable("show ip interface brief")
pprint(show_ip_int)
#### For arista switch 4
for k,v in arista_4.items():
    if k == 'data':
        intf_vars = arista_4[k]
    else:
        connect_dict[k] = arista_4[k]
connection = pyeapi.client.connect(**connect_dict,password=getpass())
device = pyeapi.client.Node(connection)
output = j2_template.render(**intf_vars)
config = (output.strip('/n')).split('\n')
cfg = config[1:3]
out = device.config(cfg)
print(out)
show_ip_int = device.enable("show ip interface brief")
pprint(show_ip_int)
