from amaranth.sim import Simulator, Settle
from program_counter import ProgramCounter

dut = ProgramCounter()

def bench():
    yield dut.countEnable.eq(0)
    yield dut.writeAdd.eq(0)
    yield dut.writeEnable.eq(0)
    yield dut.dataIn.eq(0)
    yield

    yield dut.writeEnable.eq(1)
    yield dut.dataIn.eq(1000)
    yield
    yield dut.writeEnable.eq(0)
    yield dut.writeAdd.eq(0)
    yield

    assert((yield dut.dataOut) == 1000)

    yield dut.writeEnable.eq(1)
    yield dut.writeAdd.eq(1)
    yield dut.dataIn.eq(-50)
    yield
    yield dut.writeEnable.eq(0)
    yield dut.writeAdd.eq(0)
    yield

    assert((yield dut.dataOut) == 946)

    for i in range(16):
        yield dut.countEnable.eq(1)
        yield
        assert((yield dut.dataOut) == 946 + (i*4))



sim = Simulator(dut)
sim.add_clock(1e-6) # 1 MHz
sim.add_sync_process(bench)
with sim.write_vcd("program_counter.vcd"):
    sim.run()
