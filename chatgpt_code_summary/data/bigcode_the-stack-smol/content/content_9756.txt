import os
import sys
sys.path.append(os.path.dirname(__file__))

class AbstractSystemMeter:
    """Common system meter interface for all resource monitorings.

    For each system resource to monitor, a wrapper class will be written as subclass of this one. This way we have
    a common "interface" for all system resources to test.

    This approach is choosen since python has no real interfaces like Java or C-Sharp.
    """

    def __init__(self, resource_name):
        self.resource_name = resource_name


    def measure(self, func):
        self._start()
        func()
        return self._stop()


    def _start(self):
        raise NotImplementedError("The method is not implemented yet.")


    def _stop(self):
        raise NotImplementedError("The method is not implemented yet.")
