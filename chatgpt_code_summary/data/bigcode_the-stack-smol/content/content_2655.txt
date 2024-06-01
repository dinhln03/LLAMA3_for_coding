import pyopencl as cl


class DeviceInfo(object):
    def __init__(self, device):
        self.compute_units = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
        self.maxShared = device.get_info(cl.device_info.LOCAL_MEM_SIZE) // 1024
        self.compute_capability = (
            device.get_info(cl.device_info.COMPUTE_CAPABILITY_MAJOR_NV),
            device.get_info(cl.device_info.COMPUTE_CAPABILITY_MINOR_NV)
        )
        self.deviceName = device.get_info(cl.device_info.NAME)
        self.deviceSimpleName = self.deviceName.replace(
            'GeForce', '').replace('GTX', '').strip().replace(' ', '').lower()

        print('deviceName', self.deviceName, 'compute capability', self.compute_capability)
        print('compute units', self.compute_units, 'max shared memory', self.maxShared)

        self.shared_memory_per_sm = None
        # data comes from http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls
        if self.compute_capability[0] == 5:
            if self.compute_capability[1] == 0:
                self.shared_memory_per_sm = 65536
            elif self.compute_capability[1] == 2:
                self.shared_memory_per_sm = 98304
            else:
                raise Exception('compute capability %s not recognized' % compute_capability)
        else:
            raise Exception('compute capability %s not recognized' % compute_capability)
        assert self.shared_memory_per_sm is not None

