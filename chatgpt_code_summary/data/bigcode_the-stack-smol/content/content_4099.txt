# parse list of objects
import csv

file = "basic_objects.txt"

objects = []

with open(file) as f:
    for line in f:
        if line[0:2] == '//' or line[0:2] == None:          # skip empties, comments
            pass
        else:
            obj = line.rstrip()                             # strip Newlines
            obj = obj.capitalize()
            o = [o.capitalize() for o in obj.split(' ')]    # Capitalize every word
            obj = ' '.join(o)
            objects.append(obj)

nice_objects = sorted(set(objects))
print(nice_objects[0:10], "............", nice_objects[-10:-1])
print(len(nice_objects))

def print_dupes(objs):                  # test for dupes
    last = None
    for n in nice_objects:
        if n == last:
            print(n)
        last = n

# Write out txt list
# with open("cleaned_basic_objects.txt", 'wb') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(n.split(',') for n in nice_objects)    # idk why you need the split
