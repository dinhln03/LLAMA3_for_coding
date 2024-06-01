#!/usr/bin/python
"""
A Python program that creates a list. One of the elements of the list should be
a dictionary with at least two keys. Write this list out to a file using both
YAML and JSON formats. The YAML file should be in the expanded form.
"""
import yaml
import json

a = {
'name': 'router1',
'ip_addr': '1.2.3.4',
'serial_number': 'FTX000232',
'os_version': '12.4.15T',
'optional_attrib_1': 'foo',
}

b = {
'name': 'router2',
'ip_addr': '5.6.7.8',
'serial_number': 'FTX345632',
'os_version': '12.4.15T',
}

example_list = [a, b, "empty1", "empty2"]

print "Here is the list"
print "----------------"
print example_list
print "----------------\n"
print "Here is the list in YAML"
print "------------------------"
print yaml.dump(example_list, default_flow_style=False)
print "------------------------"
print "Here is the list in JSON"
print "------------------------"
print json.dumps(example_list)
print "------------------------"

with open("example_yaml.yml", "w") as f:
    f.write(yaml.dump(example_list, default_flow_style=False))

with open("example_json.json", "w") as f:
    f.write(json.dumps(example_list))
