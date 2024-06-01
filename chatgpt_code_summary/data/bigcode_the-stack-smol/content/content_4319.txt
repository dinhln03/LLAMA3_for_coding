class Argument(object):

    def __init__(self, argument = None, base: bool = False):
        self.arg = argument
        self.is_base = base

    def __repr__(self):
        return self.arg

    def __str__(self):
        return self.arg

    def is_pipe(self):
        return self.arg == ">>" or self.arg == "<<"
