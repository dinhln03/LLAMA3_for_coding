# MIT License

# Copyright (c) 2020 Andrew Wells

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

print_LTL_game = True

obstacle_cells = [[3,0], [3,1], [3,3], [3,4], [3,5], [3,6], [3,7], [3,8], [5,2], [5,3], [5,4], [5,5], [5,6], [5,7], [5,8], [5,9], [7,0], [7,1], [7,3], [7,4], [7,5], [7,6], [7,7], [7,8]]


num_rows = 10
num_cols = 10

probN = 0.69
probE = 0.1
probW = 0.1
probB = 0.01
probS = 0.1

def rc2i_short(row, col):
    if row < num_rows and row >= 0 and col < num_cols and col >= 0:
        return row * num_rows + col
    return -1

def rc2i(row, col):
    cell = -1
    if row < num_rows and row >= 0 and col < num_cols and col >= 0:
        cell = row * num_rows + col
    for c in obstacle_cells:
        if cell == rc2i_short(c[0], c[1]):
            return -1
    return cell

def printNorth(row, col):
    extraProb = 0
    str = "[] x={} -> ".format(rc2i(i,j))
    if(rc2i(i-1, j) == -1):
        extraProb += probN
    else:
        str = str + " {}:(x'={}) +".format(probN, rc2i(i-1, j))
    if(rc2i(i+1, j) == -1):
        extraProb = extraProb + probB
    else:
        str = str + " {}:(x'={}) +".format(probB, rc2i(i+1, j))
    if(rc2i(i, j+1) == -1):
        extraProb = extraProb + probE
    else:
        str = str + " {}:(x'={}) +".format(probE, rc2i(i, j+1))
    if(rc2i(i, j-1) == -1):
        extraProb = extraProb + probW
    else:    
        str = str + " {}:(x'={}) +".format(probW, rc2i(i, j-1))
    print(str + " {}:(x'={});".format(probS+extraProb, rc2i(i,j)))

def printSouth(row, col):
    extraProb = 0
    str = "[] x={} -> ".format(rc2i(i,j))
    if(rc2i(i-1, j) == -1):
        extraProb += probB
    else:
        str = str + " {}:(x'={}) +".format(probB, rc2i(i-1, j))
    if(rc2i(i+1, j) == -1):
        extraProb = extraProb + probN
    else:
        str = str + " {}:(x'={}) +".format(probN, rc2i(i+1, j))
    if(rc2i(i, j+1) == -1):
        extraProb = extraProb + probW
    else:
        str = str + " {}:(x'={}) +".format(probW, rc2i(i, j+1))
    if(rc2i(i, j-1) == -1):
        extraProb = extraProb + probE
    else:    
        str = str + " {}:(x'={}) +".format(probE, rc2i(i, j-1))
    print(str + " {}:(x'={});".format(probS+extraProb, rc2i(i,j)))

def printEast(row, col):
    extraProb = 0
    str = "[] x={} -> ".format(rc2i(i,j))
    if(rc2i(i-1, j) == -1):
        extraProb += probW
    else:
        str = str + " {}:(x'={}) +".format(probW, rc2i(i-1, j))
    if(rc2i(i+1, j) == -1):
        extraProb = extraProb + probE
    else:
        str = str + " {}:(x'={}) +".format(probE, rc2i(i+1, j))
    if(rc2i(i, j+1) == -1):
        extraProb = extraProb + probN
    else:
        str = str + " {}:(x'={}) +".format(probN, rc2i(i, j+1))
    if(rc2i(i, j-1) == -1):
        extraProb = extraProb + probB
    else:    
        str = str + " {}:(x'={}) +".format(probB, rc2i(i, j-1))
    print(str + " {}:(x'={});".format(probS+extraProb, rc2i(i,j)))

def printWest(row, col):
    extraProb = 0
    str = "[] x={} -> ".format(rc2i(i,j))
    if(rc2i(i-1, j) == -1):
        extraProb += probE
    else:
        str = str + " {}:(x'={}) +".format(probE, rc2i(i-1, j))
    if(rc2i(i+1, j) == -1):
        extraProb = extraProb + probW
    else:
        str = str + " {}:(x'={}) +".format(probW, rc2i(i+1, j))
    if(rc2i(i, j+1) == -1):
        extraProb = extraProb + probB
    else:
        str = str + " {}:(x'={}) +".format(probB, rc2i(i, j+1))
    if(rc2i(i, j-1) == -1):
        extraProb = extraProb + probN
    else:    
        str = str + " {}:(x'={}) +".format(probN, rc2i(i, j-1))
    print(str + " {}:(x'={});".format(probS+extraProb, rc2i(i,j)))


print("mdp")
print("")
print("module M1")
print("")
if print_LTL_game:
    print("    x : [0..{}] init 0;".format(num_rows*num_cols))
else:
    print("    x : [0..{}] init 0;".format(num_rows*num_cols-1))

#print inner cells
for i in range (num_rows):
    for j in range (num_cols):
        ##Moving north
        printNorth(i,j)
        printSouth(i,j)
        printEast(i,j)
        printWest(i,j)

if print_LTL_game:
    print("")
    for i in range (num_rows*num_cols):
        print("[] x={} -> 1:(x'={});".format(i, num_rows*num_cols))
    print("[] x={} -> 1:(x'={});".format(num_rows*num_cols, num_rows*num_cols))

print("")
print("endmodule")

print("")
print("// labels")
print("label \"initial\" = (x=0);")
print("label \"loca\" = (x=26);")
print("label \"locb\" = (x=85);")
print("label \"locc\" = (x=16);")
print("label \"locd\" =  (x=7);")
print("label \"loce\" = (x=45);")
print("label \"locf\" = (x=91);")
print("label \"locg\" = (x=41);")
print("label \"loch\" =  (x=67);")
print("label \"loci\" =  (x=20);")
print("label \"zbad\" = (x=2);")
print("label \"done\" = (x={});".format(num_rows*num_cols))