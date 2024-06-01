#!/usr/bin/env python
# Decodes a given Caesar Cipher encoded string, provided on stdin
# Credit: http://stackoverflow.com/a/10792382

import sys
import string

# Frequency of each character (English)
frequency = dict(zip(string.ascii_uppercase,
                        [.0817,.0149,.0278,.0425,.1270,.0223,.0202,
                         .0609,.0697,.0015,.0077,.0402,.0241,.0675,
                         .0751,.0193,.0009,.0599,.0633,.0906,.0276,
                         .0098,.0236,.0015,.0197,.0007]))
              
# Create 26 translation tables, one for each rotation 0 through 25
# eg: ABCDEFGHIJKLMNOPQRSTUVWXYZ, BCDEFGHIJKLMNOPQRSTUVWXYZA...
trans_tables = [ string.maketrans(string.ascii_uppercase,
                 string.ascii_uppercase[i:]+string.ascii_uppercase[:i])
                 for i in range(26)] 


def fitness(msg):
  # Sum all the frequencies of each character in a string
  return sum(frequency[char] for char in msg)
  
  
def all_shifts(msg):
  # Try every rotation using the translation tables generated earlier
  # Returns a generator with three values
  msg = msg.upper()
  for index, table in enumerate(trans_tables):
    output = msg.translate(table)
    yield fitness(output), index, output
          

# Main code - accept input from stdin, find rotation with highest fitness value      
ciphertext = raw_input().replace(" ", "")
(score, index, output) = max(all_shifts(ciphertext))

print "Rotation by {:d} (key {:d}) yields decoded text:\n{}".format(index, 26-index, output)
