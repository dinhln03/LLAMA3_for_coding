"Plugin registration"
from pylint.lint import PyLinter

from .checkers import register_checkers
from .suppression import suppress_warnings


def register(linter: PyLinter) -> None:
    "Register the plugin"
    register_checkers(linter)
    suppress_warnings(linter)
