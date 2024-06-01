#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import os
import glob
import fnmatch

from echolect.core.indexing import (find_index, slice_by_value, wrap_check_start,
                                    wrap_check_stop)

from . import raw_parsing

__all__ = ['file_times', 'find_files', 'find_files_recursive', 'map_file_blocks',
           'read_voltage',
           'voltage_reader']

#******** See raw_parsing.py for details on Jicamarca raw data format ***************

def find_files(fdir, pattern='D*.r'):
    files = glob.glob(os.path.join(fdir, pattern))
    files.sort()
    return np.asarray(files)

def find_files_recursive(fdir, pattern='D*.r'):
    files = []
    for dirpath, dirnames, filenames in os.walk(fdir):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(dirpath, filename))
    files.sort()
    return np.asarray(files)

def file_times(files):
    file_times = []
    # find time that begins each file
    for fpath in files:
        with open(fpath, 'rb') as f:
            h = raw_parsing.read_first_header(f)
            time = raw_parsing.parse_time(h)
            file_times.append(time)

    return np.asarray(file_times)

def map_file_blocks(fpath):
    # read all of the headers, block times, and location for start of each data block
    headers = []
    block_times = []
    data_start_bytes = []
    with open(fpath, 'rb') as f:
        # get necesary info from first header
        h = raw_parsing.read_first_header(f)
        headers.append(h)
        time = raw_parsing.parse_time(h)
        block_times.append(time)
        data_start_bytes.append(f.tell())

        block_size = h['nSizeOfDataBlock']
        while True:
            # skip over the previous block of data
            f.seek(block_size, 1)

            # read the block's header
            try:
                h = raw_parsing.read_basic_header(f)
            except EOFError:
                break

            time = raw_parsing.parse_time(h)
            # check validity of header
            # assume that if time is 0, the subsequent block was
            # not written and hence EOF has been reached
            if time == 0:
                break

            headers.append(h)
            block_times.append(time)
            data_start_bytes.append(f.tell())

    headers = np.asarray(headers)
    block_times = np.asarray(block_times)
    data_start_bytes = np.asarray(data_start_bytes)

    return headers, block_times, data_start_bytes

class voltage_reader(object):
    def __init__(self, fpath):
        self.fpath = fpath
        headers, block_times, data_start_bytes = map_file_blocks(fpath)
        self.headers = headers
        self.block_times = block_times
        self.data_start_bytes = data_start_bytes

        h = headers[0]
        raw_dtype, dtype = raw_parsing.parse_dtype(h)
        self.raw_dtype = raw_dtype
        self.dtype = dtype

        block_shape = raw_parsing.parse_block_shape(h)
        self.block_shape = block_shape
        self.nprofiles_per_block = block_shape[0]
        self.nsamples_per_profile = block_shape[1]
        self.nchannels = block_shape[2]
        self.nitems_per_profile = block_shape[1]*block_shape[2]
        self.nblocks = len(block_times)
        self.nprofiles = self.nblocks*block_shape[0]
        self.profile_bytes = self.nitems_per_profile*self.raw_dtype.itemsize
        self.shape = (self.nprofiles, self.nchannels, self.nsamples_per_profile)

        self.ts = raw_parsing.parse_ts(h)
        self.ipp = raw_parsing.parse_ipp(h)
        self.r = raw_parsing.parse_range_index(h)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            pidx = key
            cidx = slice(None)
            sidx = slice(None)
        else:
            lkey = len(key)
            if lkey < 1 or lkey > 3:
                raise IndexError('Wrong number of indices')
            elif lkey == 1:
                pidx = key[0]
                cidx = slice(None)
                sidx = slice(None)
            elif lkey == 2:
                pidx = key[0]
                cidx = key[1]
                sidx = slice(None)
            else:
                pidx = key[0]
                cidx = key[1]
                sidx = key[2]

        if isinstance(pidx, int):
            return self.read_voltage(pidx, chan_idx=cidx, sample_idx=sidx)

        # pidx must be a slice object
        start, stop, step = pidx.indices(self.shape[0])
        return self.read_voltage(start, stop, step, cidx, sidx)

    def _read_from_block(self, block_num, start, stop, step, chan_idx, sample_idx):
        num = stop - start
        # so we drop dimension when num == 1
        if num == 1:
            prof_idx = 0
        else:
            prof_idx = slice(0, num, step)

        fstart = self.data_start_bytes[block_num] + self.profile_bytes*start
        with open(self.fpath, 'rb') as f:
            f.seek(fstart)
            vlt_raw = np.fromfile(f, self.raw_dtype, num*self.nitems_per_profile)
        # data arranged by channel, then sample, then profile
        try:
            vlt_raw = vlt_raw.reshape(num, self.nsamples_per_profile, self.nchannels)
        except ValueError: # we didn't get the number of samples we expected, reshape fails
            raise EOFError('End of file reached. Could not read requested data.')
        # swap axes so it matches what we want, and slice as desired
        vlt_raw = vlt_raw.swapaxes(1, 2)[prof_idx, chan_idx, sample_idx]
        vlt = np.empty(vlt_raw.shape, dtype=self.dtype)
        vlt.real = vlt_raw['real']
        vlt.imag = vlt_raw['imag']

        return vlt

    def read_from_block(self, block_num, start, stop=None, step=1, nframes=1,
                        chan_idx=slice(None), sample_idx=slice(None)):
        start = wrap_check_start(self.nprofiles_per_block, start)
        if stop is None:
            stop = start + step*nframes
        else:
            stop = wrap_check_stop(self.nprofiles_per_block, stop)
        ## change ints to lists so that we don't lose dimensions when indexing
        #if isinstance(chan_idx, int):
            #chan_idx = [chan_idx]
        #if isinstance(sample_idx, int):
            #sample_idx = [sample_idx]
        return self._read_from_block(block_num, start, stop, step, chan_idx, sample_idx)

    def _read_from_blocks(self, blocknumstart, blockstart, blocknumend, blockstop,
                          step, chan_idx, sample_idx):
        if blocknumstart == blocknumend:
            return self.read_from_block(blocknumstart, blockstart, blockstop, step,
                                        chan_idx, sample_idx)

        start = blockstart
        vlt_all = []
        for bnum in xrange(blocknumstart, blocknumend + 1):
            if bnum == blocknumend:
                vlt_all.append(self.read_from_block(bnum, start, blockstop, step,
                                                    chan_idx, sample_idx))
            else:
                vlt_all.append(self.read_from_block(bnum, start, self.nprofiles_per_block, step,
                                                    chan_idx, sample_idx))
                # set start for (possibly != 0) based on step
                # step - ((nprofiles - start) % step) == (start - nprofiles) % step
                start = (start - self.nprofiles_per_block) % step

        return np.concatenate(vlt_all, axis=0)

    def read_voltage(self, start, stop=None, step=1, nframes=1,
                     chan_idx=slice(None), sample_idx=slice(None)):
        start = wrap_check_start(self.shape[0], start)
        if stop is None:
            stop = start + step*nframes
        else:
            stop = wrap_check_stop(self.shape[0], stop)
        ## change ints to lists so that we don't lose dimensions when indexing
        #if isinstance(chan_idx, int):
            #chan_idx = [chan_idx]
        #if isinstance(sample_idx, int):
            #sample_idx = [sample_idx]

        # find blocks for start and stop
        bstart, strt = divmod(start, self.nprofiles_per_block)
        # want block of last profile to include, hence profile number end = stop - 1
        end = stop - 1
        bend, nend = divmod(end, self.nprofiles_per_block)
        stp = nend + 1

        return self._read_from_blocks(bstart, strt, bend, stp, step, chan_idx, sample_idx)

def read_voltage(fpath, key=slice(None)):
    vlt_r = voltage_reader(fpath)
    return vlt_r[key]
