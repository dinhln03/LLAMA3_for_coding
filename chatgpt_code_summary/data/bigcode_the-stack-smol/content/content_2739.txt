# Copyright (C) 2019 by Landmark Acoustics LLC
r"""A class to write a WAV-formatted file."""

import wave


class WaveFile:
    '''A wrapper for `Wave_write` from Python STL's `wave` module.

    Parameters
    ----------
    name : str
        The name to save the file as. It should include path and extension.

    sample_rate : int
        The number of samples per second that the file will use.

    bit_rate : int
        The number of bits the file will use per sample.

    channels : int
        The number of channels that the file has.

    See Also
    --------
    wave : the Python STL module

    '''

    def __init__(self,
                 name: str,
                 sample_rate: int,
                 bit_rate: int,
                 channels: int) -> None:
        self._channels = channels
        self._sample_rate = sample_rate
        self._byte_rate = bit_rate // 8
        self._filehandle = wave.open(name, 'wb')
        self._filehandle.setnchannels(self.channels)
        self._filehandle.setsampwidth(self.byte_rate)
        self._filehandle.setframerate(self.sample_rate)

    @property
    def channels(self) -> int:
        '''The number of channels the file has.'''
        return self._channels

    @property
    def sample_rate(self) -> int:
        '''The number of samples per second.'''
        return self._sample_rate

    @property
    def byte_rate(self) -> int:
        '''The number of bytes per sample.'''
        return self._byte_rate

    @property
    def bit_rate(self) -> int:
        '''The number of bits per sample.'''
        return self.byte_rate * 8

    def write_frames(self, data) -> int:
        '''Add some data to the file.

        Parameters
        ----------
        data : bytes-like object
            The user must ensure that the data's format matches the file's!

        Returns
        -------
        int : the number of frames written

        '''

        pos = self._filehandle.tell()
        self._filehandle.writeframes(data)
        return self._filehandle.tell() - pos

    @property
    def frame_size(self) -> int:
        '''The number of bytes per frame.'''

        return self.byte_rate * self.channels

    def __enter__(self):
        self._filehandle.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        return self._filehandle.__exit__(*args, **kwargs)


if __name__ == '__main__':
    import array
    import sys
    wvf = WaveFile(sys.argv[1], 44100, 28, 3)
    a = array.array('b')
    a.extend([0 for i in range(12000 * wvf.frame_size)])
    N = wvf.write_frames(a)
    print(f'Wrote {N} frames in {wvf.channels} {wvf.bit_rate}-bit channels.')
