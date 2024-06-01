import os
import sys
import subprocess
from subprocess import CalledProcessError
from subprocess import TimeoutExpired
import time
import re
import statistics

class MemPoint:
    def __init__(self, time, heap, heap_extra, stack, heap_tree):
        self.time = int(time.split("=")[1])
        self.heap = int(heap.split("=")[1])
        self.heap_extra = int(heap_extra.split("=")[1])
        self.stack = int(stack.split("=")[1])

    def get_sum_memusage(self):
        return self.heap + self.heap_extra + self.stack

def get_mem_usage(filename):
    with open(filename) as file:
        contents = file.readlines()
    memory_points = []
    for index in range(len(contents)):
        if("snapshot" in contents[index]):
            emptyLine = contents[index+1] # not used
            time = contents[index+2] # not used
            mem_heap = contents[index+3]
            mem_heap_extra = contents[index+4]
            mem_stacks = contents[index+5]
            heap_tree = contents[index+6] #not used
            memory_points.append(MemPoint(time, mem_heap, mem_heap_extra, mem_stacks, heap_tree))
    maxUsage = max(value.get_sum_memusage() for value in memory_points)
    return maxUsage

def getFileSize(filename):
    return os.path.getsize(filename)


def purge(dir):
    for f in os.listdir(dir):
        if "massif.out." in f:
            os.remove(os.path.join(dir, f))


def getRam(matrix, filename):
    purge(".")
    subprocess.run(["valgrind", "--tool=massif", "--stacks=yes",
                    "--pages-as-heap=no", filename, matrix])
    ps = subprocess.Popen(('ls'), stdout=subprocess.PIPE)
    resultFilename = subprocess.check_output(
        ('grep', 'massif.out'), stdin=ps.stdout).decode(sys.stdout.encoding).strip()
    ps.wait()
    maxUsage = get_mem_usage(resultFilename)
    return maxUsage

def getSpeed(matrix, filename):
    start = time.time()
    resultLines = subprocess.check_output((filename, matrix), stderr=subprocess.STDOUT, timeout=5).decode(sys.stdout.encoding)
    end = time.time() - start
    return end

def getSolutionFromString(string):
    string = string.split('\n')
    string = [x.strip() for x in string]
    string = list(filter(lambda a: a != '', string))
    if len(string) != 2:
        raise Exception("Wrong number of lines in outout")
    output = {}
    output["rows"] = [int(x) for x in string[0].split()]
    output["score"] = int(string[1])
    return output

def isValid(matrix, filename, solutionFilename):
    try:
        resultLines = subprocess.check_output((filename, matrix), stderr=subprocess.STDOUT, timeout=5).decode(sys.stdout.encoding)
    except CalledProcessError as e:
        print("The application did not exit cleanly on: " + matrix)
        return False
    except TimeoutExpired as e:
        print("The application exceeded the allowed time on: " + matrix)
        return False
    result = getSolutionFromString(resultLines)
    with open(solutionFilename) as solutionFile:
        solutionLines = solutionFile.read()
    solution = getSolutionFromString(solutionLines)
    return solution == result
    for i in range(len(resultLines)):
        if resultLines[i] != solutionLines[i]:
            print('Incorrect solution.')
            print('Expected: ' + str(resultLines))
            print('Received: ' + str(solutionLines))
            return False
    return True


def tryInt(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryInt(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def printHelpInfo():
    print("")
    print("Usage: python3 path/to/executable path/to/directory/with/matrices/and/solutions")
    print("")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Wrong number of arguments!")
        printHelpInfo()
        sys.exit()
    filename = sys.argv[1]
    myDir = os.path.dirname(__file__)
    if not os.path.exists(filename):
        print("Executable file \"" + filename + "\" not found")
        printHelpInfo()
        sys.exit()
        

    directory = sys.argv[2]
    if not os.path.exists(directory) or not os.path.isdir(directory):
        print("Matrix directory \"" + directory + "\" is not a valid path")
        printHelpInfo()
        sys.exit()

    fileSize = getFileSize(filename)
    matrices = [os.path.join(directory, f) for f in os.listdir(directory) if "matrix" in f]
    matrices.sort(key=alphanum_key)
    solutions = [os.path.join(directory, f)  for f in os.listdir(directory) if "solution" in f]
    solutions.sort(key=alphanum_key)
    timedout = []
    speeds = []
    rams = []

    for index in range(len(matrices)):
        matrix = matrices[index]
        solution = solutions[index]
        valid = isValid(matrix, filename, solution)
        if not valid:
            break
        speed = getSpeed(matrix, filename)
        ram = getRam(matrix, filename)
        speeds.append(speed)
        rams.append(ram)

    print("Filesize is " + str(fileSize))
    print("")
    print("Speeds are " + str(speeds))
    print("")
    print("Average Speed is: " + str(statistics.mean(speeds)))
    print("")
    print("Rams are " + str(rams))
    print("")
    print("Average Ram is: " + str(round(statistics.mean(rams))))
