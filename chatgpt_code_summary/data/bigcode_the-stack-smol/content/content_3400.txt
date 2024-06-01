#!/usr/bin/env python

# import general use modules
import os
from pprint import pprint as pp
# import nornir specifics
from nornir import InitNornir
from nornir.plugins.functions.text import print_result
from nornir.core.filter import F

nr = InitNornir()
hosts = nr.inventory.hosts
arista1_filter = nr.filter(name="arista1")
arista1 = arista1_filter.inventory.hosts

#print(hosts)
print(arista1)

wan_filter = nr.filter(role="WAN")
wan_filter = wan_filter.inventory.hosts

print(wan_filter)

wan_port_filter = nr.filter(role="WAN").filter(port=22)
wan_port_filter = wan_port_filter.inventory.hosts

print(wan_port_filter)

sfo_filter = nr.filter(F(groups__contains="sfo"))
sfo_filter = sfo_filter.inventory.hosts

print(sfo_filter)

