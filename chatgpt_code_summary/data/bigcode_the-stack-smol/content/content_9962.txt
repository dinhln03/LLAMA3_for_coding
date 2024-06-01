from ..commands.help import HelpCommand
from ..commands.exit import ExitCommand
from ..commands.purchase import PurchaseCommand

class CommandState:
    """
        The __state value should not be accessed directly,
        instead the get() method should be used.
    """

    __state = {
        'commands': {
            'help': HelpCommand.execute,
            'exit': ExitCommand.execute,
            'purchase': PurchaseCommand.execute,
        },
    }

    @classmethod
    def get(cls, key):
        return cls.__state.get(key)