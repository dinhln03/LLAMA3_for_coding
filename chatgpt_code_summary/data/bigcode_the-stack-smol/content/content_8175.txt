
# MIT License
# 
# Copyright (c) 2019 Meyers Tom
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import fileinput
import runner.config

def exclude(list, value=["-v", "-vv", "-vvv"]):
    """
    remove value from the list
    """
    new = []
    for item in list:
        if not item in value:
            new.append(item)
    return new

def help():
    """
    print the help menu and exit
    """
    print("Usage: {} [-h | --help] [-vv] domain".format(runner.config.NAME))
    print("\t {} -h | --help \t\tprint this help menu".format(runner.config.NAME))
    print("\t {} -v <domain> \t\tEnable debug messages".format(runner.config.NAME))
    print("\t {} -vv <domain> \t\tEnable debug messages with more information".format(runner.config.NAME))
    print("\t {} <domain> \t\tperform a scan on the given domain/URI or URL\n".format(runner.config.NAME))
    print("Copyright Meyers Tom")
    print("Licensed under the MIT License")
    quit()

def getInput():
    """
    Check if the input is null. If that is the case simply listen for stdin
    Returns the input that it got eg a url, uri or domain
    Second return type is if debug messages should be enabled
    """
    if len(sys.argv) == 1:
        return fileinput.input()[0], False, False
    if "-h" in sys.argv or "--help" in sys.argv:
        help()
    domain = "".join(exclude(sys.argv[1:], ["-v", "-vv"]))
    if domain == "":
        print("Wrong input formation\n")
        help()
    return domain, "-v" in sys.argv, "-vv" in sys.argv