import json

from symro.src.automenu import Command


class SpecialCommand(Command):

    def __init__(self,
                 symbol: str,
                 line_index: int = -1):
        super(SpecialCommand, self).__init__()
        self.symbol: str = symbol
        self.line_index: int = line_index

    def __str__(self) -> str:

        arg_tokens = []
        for arg in self.get_ordered_args():
            arg_tokens.append(str(arg))
        for name, value in self.get_named_args().items():
            if isinstance(value, list) or isinstance(value, dict):
                value = json.dumps(value)
            arg_tokens.append("{0}={1}".format(name, value))

        arg_str = ""
        if len(arg_tokens) > 0:
            arg_str = "(" + ", ".join(arg_tokens) + ")"

        return "@{0}{1}".format(self.symbol, arg_str)
