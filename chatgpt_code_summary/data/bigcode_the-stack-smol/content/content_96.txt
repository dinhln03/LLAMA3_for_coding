#!/usr/bin/python
import argparse
import subprocess
import json

parser = argparse.ArgumentParser()
parser.add_argument("--list", action="store_true")
args = parser.parse_args()

result = {"_meta": {"hostvars": {}}}
if args.list:
    output = subprocess.check_output([
        "cd ../terraform/stage; terraform show -json"
    ], shell=True)
    data = json.loads(output)

    group_list = set()

    try:
        for module in data["values"]["root_module"]["child_modules"]:
            try:
                for resource in module["resources"]:
                    if resource["type"] == "null_resource":
                        continue
                    group_name = resource["name"]
                    values = resource["values"]
                    host_name = values["name"]
                    ip = values["network_interface"][0]["nat_ip_address"]

                    if group_name not in result:
                        result[group_name] = {"hosts": []}

                    group_list.add(group_name)

                    result[group_name]["hosts"].append(host_name)
                    result["_meta"]["hostvars"][host_name] = {
                        "ansible_host": ip
                    }
            except KeyError:
                continue

        result["all"] = {"children": list(group_list), "hosts": [], "vars": {}}
    except KeyError:
        pass

    print(json.dumps(result))

else:
    print(json.dumps(result))
