import json
from videomaker.functions.packageData import packageData
from videomaker.functions.addPreset import addOption

def savePreset(focus):
    preset = packageData(focus, verify=False)
    with open("./presets/{0}.json".format(preset["subredditName"]), "w+") as out:
        json.dump(preset, out, indent=4)
    addOption(focus, "./presets/{0}.json".format(preset["subredditName"]), preset["subredditName"])
