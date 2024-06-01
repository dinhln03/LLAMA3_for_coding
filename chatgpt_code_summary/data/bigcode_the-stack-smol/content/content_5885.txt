import ctypes
import pytest

c_lib = ctypes.CDLL('../solutions/0709-to-lower/to-lower.so')

@pytest.mark.parametrize('string, ans',
                        [(b"Hello", b"hello"),
                         (b"here", b"here"),
                         (b"LOVELY", b"lovely")])
def test_to_lower(string, ans):
    c_lib.toLowerCase(string)
    assert string == ans
