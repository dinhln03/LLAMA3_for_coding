import sys

import pytest

from organizer.util.arguments import is_option


@pytest.mark.parametrize("arg", ['verbose', 'silent', 'dry-run', 'ignored', 'exists'])
def test_have_arguments(arg: str):
    sys.argv = ['--' + arg]
    assert is_option(arg)


@pytest.mark.parametrize("arg", ['verbose', 'silent', 'dry-run', 'ignored', 'exists'])
def test_no_have_arguments(arg: str):
    sys.argv = []
    assert not is_option(arg)
