def test_positive_guess(patched_hangman):
    decision = patched_hangman.guess("e")
    assert decision is True


def test_negative_guess(patched_hangman):
    decision = patched_hangman.guess("r")
    assert decision is False


def test_none_guess(patched_hangman):
    patched_hangman.guess("e")
    decision = patched_hangman.guess("e")
    assert decision is None
