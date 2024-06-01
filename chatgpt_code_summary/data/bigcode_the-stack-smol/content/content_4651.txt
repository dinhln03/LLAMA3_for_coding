# This file is part of postcipes
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .postcipe import Postcipe
import turbulucid as tbl
import numpy as np
import h5py

__all__ = ["BackwardFacingStep"]


class BackwardFacingStep(Postcipe):

    def __init__(self, path, nu, uRef):
        Postcipe.__init__(self)
        self.case = tbl.Case(path)
        self.nu = nu
        self.uRef = uRef
        self.h = np.sum(tbl.edge_lengths(self.case, "stepB"))
        self.H = np.sum(tbl.edge_lengths(self.case, "outletB")) - self.h
        self.eRatio = (self.H + self.h)/self.H

        self.tau1 = \
            self.case.boundary_data("lowB1")[1]["wallShearStressMean"][:, 0]
        self.tau2 = \
            self.case.boundary_data("lowB2")[1]["wallShearStressMean"][:, 0]
        self.tau = np.append(self.tau1, self.tau2)

        self.x1 = self.case.boundary_data("lowB1")[0][:, 0]
        self.x2 = self.case.boundary_data("lowB2")[0][:, 0]
        self.x = np.append(self.x1, self.x2)

        self.idx105h = np.argmin(np.abs(self.x1 + 1.05*self.h))

        self.uTop = self.case.boundary_data("upB")[1]['UMean'][:, 0]

        self.theta = None
        self.delta99 = None
        self.edgeU = None



    def compute_delta99(self, u0='max', interpolate=True):
        self.delta99 = np.zeros(self.x1.shape[0])
        self.edgeU = np.zeros(self.x1.shape[0])

        for i in range(self.x1.shape[0]):
            x = self.x1[i]
            y, v = tbl.profile_along_line(self.case, (x, -1), (x, 10),
                                          correctDistance=True)
            self.delta99[i] = tbl.delta_99(y, v['UMean'][:, 0], u0=u0,
                                           interpolate=interpolate)
            if u0 is 'max':
                self.edgeU[i] = np.max(v['UMean'][:, 0])
            elif u0 is 'last':
                self.edgeU[i] = v['UMean'][-1, 0]


        self.reDelta99 = self.delta99*self.edgeU/self.nu
        self.reTau = self.delta99*np.sqrt(np.abs(self.tau1))/self.nu
        self.delta99105h = self.delta99[self.idx105h]
        return 0

    def compute_theta(self, u0='max', interpolate=True):
        self.theta = np.zeros(self.x1.shape[0])
        self.edgeU = np.zeros(self.x1.shape[0])

        for i in range(self.x1.shape[0]):
            x = self.x1[i]
            y, v = tbl.profile_along_line(self.case, (x, -1), (x, 10),
                                          correctDistance=True)
            self.theta[i] = tbl.momentum_thickness(y, v['UMean'][:, 0], u0=u0,
                                                   interpolate=interpolate)
            if u0 is 'max':
                self.edgeU[i] = np.max(v['UMean'][:, 0])
            elif u0 is 'last':
                self.edgeU[i] = v['UMean'][-1, 0]

        self.reTheta = self.theta*self.edgeU/self.nu
        self.reTheta105h = self.reTheta[self.idx105h]
        return 0

    def save(self, name):
        f = h5py.File(name, 'w')

        f.attrs["h"] = self.h
        f.attrs["H"] = self.H
        f.attrs["nu"] = self.nu
        f.attrs["eRatio"] = self.eRatio
        f.attrs["uRef"] = self.uRef
        f.attrs["idx105h"] = self.idx105h

        f.create_dataset("x1", data=self.x1)
        f.create_dataset("x2", data=self.x2)
        f.create_dataset("x", data=self.x)

        f.create_dataset("uTop", data=self.uTop)

        f.create_dataset("tau1", data=self.tau1)
        f.create_dataset("tau2", data=self.tau2)
        f.create_dataset("tau", data=self.tau)

        if self.theta is None:
            self.compute_theta()
        if self.delta99 is None:
            self.compute_delta99()
        f.create_dataset("theta", data=self.theta)
        f.create_dataset("delta99", data=self.delta99)
        f.create_dataset("reTheta", data=self.reTheta)
        f.create_dataset("reTau", data=self.reTau)
        f.create_dataset("reDelta99", data=self.reDelta99)

        f.close()

    def load(self, name):
        f = h5py.File(name, 'r')

        self.h = f.attrs["h"]
        self.H = f.attrs["H"]
        self.nu = f.attrs["nu"]
        self.eRatio = f.attrs["eRatio"]
        self.uRef = f.attrs["uRef"]
        self.idx105h = f.attrs["idx105h"]

        self.x1 = f["x1"][:]
        self.x2 = f["x2"][:]
        self.x = f["x"][:]

        self.uTop = f["uTop"][:]

        self.tau1 = f["tau1"][:]
        self.tau2 = f["tau2"][:]
        self.tau = f["tau"][:]

        self.theta = f["theta"][:]
        self.delta99 = f["delta99"][:]
        self.reTheta = f["reTheta"][:]
        self.reTau = f["reTau"][:]
        self.reDelta99 = f["reDelta99"][:]
        f.close()

