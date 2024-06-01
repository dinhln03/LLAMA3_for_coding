#!/bin/bash

from pkg_resources import require
require('numpy')
require('h5py')

import sys, os
import numpy as np
import h5py

cxifilenames = sys.argv[2:]

output_dims = tuple()
print "Using CXI file for dims: ", cxifilenames[0]
with h5py.File(cxifilenames[0], 'r') as cxi:
    output_dtype = cxi['entry_1']['data_1']['data'].dtype
    output_dims = (len(cxifilenames), 
                   cxi['entry_1']['data_3']['data'].shape[0] * 2, 
                   cxi['entry_1']['data_3']['data'].shape[1])



print output_dims, output_dtype

dset = np.zeros(shape=output_dims, dtype = output_dtype)

for i, cxi_file in enumerate(cxifilenames):
    with h5py.File(cxi_file, 'r') as cxi:
        print cxi_file
        cxi_dset = cxi['entry_1']['data_3']['data']
        offset = (i, 0, 0)
        print "    ", offset, cxi_dset.shape
        dset[offset[0], offset[1]:cxi_dset.shape[0]+offset[1], offset[2]:cxi_dset.shape[1]+offset[2]] = cxi_dset
        
        cxi_dset = cxi['entry_1']['data_4']['data']
        offset = (i, output_dims[1]/2, 0)
        print "    ", offset, cxi_dset.shape
        dset[offset[0], offset[1]:cxi_dset.shape[0]+offset[1], offset[2]:cxi_dset.shape[1]+offset[2]] = cxi_dset

print "Large dataset created: ", dset
print "min/max/mean value: ", dset.min(), dset.max(), dset.mean()
print "Raising data values by (turning into unsigned dataset): ", abs(dset.min())
unsigned_dset = np.array(dset + abs(dset.min()), dtype=np.uint16)

print "Creating file: ", sys.argv[1]
out = h5py.File(sys.argv[1], 'w')
print "Creating dataset in output file"
out_dset = out.create_dataset('data', data = unsigned_dset)
print "Done. Closing file"
out.close()
