from cpu_element import CPU_element
from elements import Instruction_pointer
from tests.tools import set_signals

signals = ["address"]
result = "result"


def test_write_output():
    source = CPU_element([], signals)
    ip = Instruction_pointer(signals, [result])
    assert isinstance(ip, CPU_element)
    ip.connect([source])
    value = 55
    set_signals(source, ip, signals, [value])
    assert ip.outputs[result] == value
