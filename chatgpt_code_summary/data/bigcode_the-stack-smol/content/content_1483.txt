# Simple script for updating the build number in pubspec.yaml
import re

# Regex patter to be used to ID the correct line in pubspec.yaml
version_line_pattern = "version:\s+\d+\.\d+\.\d+\+\d+"

# Open pubspec.yaml and read lines into memory
with open("pubspec.yaml", "r") as current_pubspec:
    contents = current_pubspec.readlines()

# Reopen pubspec.yaml for writing and update
with open("pubspec.yaml", "w") as updated_pubspec:
    
    # Find and bump build number
    counter = 0
    for line in contents:
        if re.match(pattern=version_line_pattern, string=line):
            line_array = line.split("+")
            contents[counter] = line_array[0] + "+" + str(int(line_array[1]) + 1) + "\n"
            break
        counter += 1
    
    # Write updated contents back to disk
    updated_pubspec.writelines(contents)
