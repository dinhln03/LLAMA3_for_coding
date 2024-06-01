from typing import Callable

import unittest

# test
from .pipe import pipe


class TestPipe(unittest.TestCase):
    def test_pipe_should_return_a_function(self) -> None:
        # given
        def echo(x: str) -> str:
            return f"echo {x}"

        # when
        output = pipe(echo)

        # then
        self.assertTrue(isinstance(output, Callable))  # type: ignore

    def test_pipe_should_return_an_empty_string(self) -> None:
        # given
        def echo(x: str) -> str:
            return f"echo {x}"

        # when
        param = "hello world"
        output = pipe(echo)(param)

        # then
        self.assertEqual(output, f"echo {param}")

    def test_pipe_should_pipe_two_function(self) -> None:
        # given
        def echo(x: str) -> str:
            return f"echo {x}"

        def grep() -> str:
            return "grep world"

        # when
        param = "hello world"
        output = pipe(echo, grep)(param)

        # then
        self.assertEqual(output, f"echo {param} | grep world")
