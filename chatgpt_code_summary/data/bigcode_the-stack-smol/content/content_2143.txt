from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class SubCommand(object):
    name = NotImplementedError("Please add 'name' member in your SubCommand")
    help = NotImplementedError("Please add 'help' member in your SubCommand")

    def addParser(self, parser):
        raise NotImplementedError("Please implement 'addParser' method in your SubCommand")

    def execute(self):
        raise NotImplementedError("Please implement 'execute' method in your SubCommand")
