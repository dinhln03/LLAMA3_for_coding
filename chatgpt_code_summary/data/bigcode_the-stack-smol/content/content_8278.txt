#! /usr/local/bin/python

# pylint: disable=invalid-name
# pylint: disable=missing-docstring

with open("table.txt", "r") as f:
    print "projections = ("
    lines = f.readlines()
    units_list = []
    for line in lines:
        lineparts = line.rstrip().split("|")
        epsg = lineparts[0]
        wkt = lineparts[3]
        name = wkt.split("\"")[1]

        location = line.find("+units=")
        if location == -1:
            unit_index = line.find("UNIT[")
            unit_code = line[unit_index:].rstrip().split("\"")[1]
            units_list.append(unit_code)
        else:
            unit_code = line[location:].rstrip().split(" ")[0].split("=")[1]

        units_list.append(unit_code)

        if unit_code == "m":
            unit = "Meter"
            unit_factor = 1
        elif unit_code == "ft":
            unit = "International Foot"
            unit_factor = 0.3048
        elif unit_code == "us-ft":
            unit = "US Survey Foot"
            unit_factor = 0.3048006096012192
        elif unit_code == "grad":
            unit = "Gradian"
            unit_factor = 0.01470796326794897
        elif unit_code == "degree":
            unit = "Degree"
            unit_factor = 0.0174532925199433
        else:
            unit = "Unknown"
            unit_factor = 0

        print "{"
        print "wkt = \"" + wkt.replace("\"", "\\\"") + "\";"
        print "name = \"" + name + "\";"
        print "unit = \"" + unit + "\";"
        print "unit_factor = " + str(unit_factor) + ";"
        print "epsg = " + str(epsg) + ";"
        print "},"

print ")"
