import cpboard
import periphery
import pytest
import smbus
import sys

def pytest_addoption(parser):
    group = parser.getgroup('i2cslave')
    group.addoption("--bus", dest='i2cbus', type=int, help='I2C bus number')
    group.addoption("--serial-wait", default=20, dest='serial_wait', type=int, help='Number of milliseconds to wait before checking board output (default: 20ms)')
    group.addoption("--smbus-timeout", default=True, dest='smbus_timeout', type=bool, help='Use SMBUS timeout limit (default: True)')


@pytest.fixture(scope='session')
def board(request):
    board = cpboard.CPboard.from_try_all(request.config.option.boarddev)
    board.open()
    board.repl.reset()
    return board


class I2C:
    def __init__(self, bus):
        self.bus = periphery.I2C('/dev/i2c-%d' % bus)

    def __enter__(self):
        return self

    def __exit__(self, t, value, traceback):
        self.close()

    def close(self):
        self.bus.close()

    def transfer(self, address, messages):
        #__tracebackhide__ = True # Hide this from pytest traceback
        self.bus.transfer(address, messages)

    Message = periphery.I2C.Message

    def read(self, address, n):
        data = [0] * n
        msgs = [I2C.Message(data, read=True)]
        self.transfer(address, msgs)
        return msgs[0].data

    def write(self, address, data):
        msgs = [I2C.Message(data)]
        self.transfer(address, msgs)

    def write_read(self, address, data, n):
        recv = [0] * n
        msgs = [I2C.Message(data), I2C.Message(recv, read=True)]
        self.transfer(address, msgs)
        return msgs[1].data


@pytest.fixture
def i2cbus(request):
    return I2C(request.config.option.i2cbus)
