import pytest
import activeLearning as tP

def test_sayHello():
    assert tP.sayHello() == 'Hello World'
    assert tP.sayHello('Sankha') == 'Hello Sankha'
    assert tP.sayHello(-1) == 'Hello -1'
    return
