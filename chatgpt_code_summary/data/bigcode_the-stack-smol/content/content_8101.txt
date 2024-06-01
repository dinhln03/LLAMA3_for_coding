#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from ..pipelineState import PipelineStateInterface
from ..data import BrewPipeDataFrame

__author__ = 'Dominik Meyer <meyerd@mytum.de>'


class NumpyNullPreprocessor(PipelineStateInterface):
    """
    This is an example class of preprocessor, that
    takes numpy data from the data loader and outputs
    numpy data again. Basically, it does nothing, and is
    just a testcase to get some interface definitions going.
    """

    def __init__(self, intermediate_directory="intermediates"):
        """
        :param intermediate_directory: Directory, where the
            intermediate pandas dataframe should be persisted
            to.
        """
        super(NumpyNullPreprocessor, self).__init__()

        self._intermediate_directory = intermediate_directory
        self._cached = False
        self._cached_object = None

    def _persist_numpy(self, arr, name):
        filename = os.path.join(self._intermediate_directory,
                                'NumpyNullPreprocessor' + name)
        with open(filename, 'w') as f:
            np.save(f, arr)
        return filename

    def _load_numpy(self, name):
        filename = os.path.join(self._intermediate_directory,
                                'NumpyNullPreprocessor' + name)
        with open(filename, 'r') as f:
            arr = np.load(f)
        return arr

    def preprocess(self, dataframe):
        def cb(name):
            obj = self
            inp = dataframe
            h = obj.get(inp.name)
            tmp = None
            if not h or h != inp.hash:
                org = inp.data
                # preprocessing would happen here and be put to tmp
                tmp = org
                h = inp.hash
                obj._persist_numpy(tmp, inp.name)
                obj.put(inp.name, h)
            else:
                if self._cached and self._cached == inp.hash:
                    return self._cached_object
                tmp = obj._load_numpy(inp.name)
                self._cached_object = tmp
                self._cached = inp.hash
            return tmp

        h = 0
        if not self.get(dataframe.name) is None:
            h = self.get(dataframe.name)
        r = BrewPipeDataFrame(dataframe.name, lazy_frame=True, hash=h, callback=cb)
        return r


