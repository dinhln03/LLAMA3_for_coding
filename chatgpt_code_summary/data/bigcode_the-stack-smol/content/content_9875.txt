import threading
from time import sleep

from .intcode import Intcode


class Amplifier(object):
    def __init__(self, mem_str: str):
        self._amps = [Intcode(mem_str, name=f'Amp {n + 1}') for n in range(5)]

    def run(self, inputs: str or list, trace=False, quiet=True):
        out = 0
        p = self._amps[0]
        if isinstance(inputs, str):
            inputs = [int(v) for v in inputs.split(',')]
        for inp in inputs:
            p.reset_core()
            p.simulate([inp, out], trace=trace)
            out = p.output[0]
        self._print_log(quiet)
        return out

    def _print_log(self, quiet):
        if not quiet:
            for p in self._amps:
                msg = f'{p.name} log:'
                top_n_tail = "*" * (len(msg) + 4)
                print(top_n_tail)
                print(f'* {msg} *')
                print(top_n_tail)
                print('\n'.join(p.get_log()))

    def run_regeneration(self, inputs: str or list, trace=False, quiet=True):
        if isinstance(inputs, str):
            inputs = [int(v) for v in inputs.split(',')]
        p = self._amps[0]
        p.reset_core()
        for n in self._amps[1:]:
            p.connect(n.receiver)
            p = n
            p.reset_core()
        self._amps[-1].connect(self._amps[0].receiver)
        threads = []
        for a, n in zip(self._amps, inputs):
            a.receiver(n)
            t = threading.Thread(target=a.simulate, kwargs={'trace': trace})
            threads.append(t)
            t.start()
        self._amps[0].receiver(0)
        while any(t.is_alive() for t in threads):
            sleep(0.0001)
        self._print_log(quiet)

        return self._amps[0]._input.pop()