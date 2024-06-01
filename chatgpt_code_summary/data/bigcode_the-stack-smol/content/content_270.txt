# -*- coding: utf-8 -*-
"""
Copyright (c) 2021 Showa Denko Materials co., Ltd. All rights reserved.

This software is for non-profit use only.

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THIS SOFTWARE OR THE USE OR OTHER DEALINGS IN THIS SOFTWARE.
"""

import time
import numpy as np
from GPyOpt.core.task.objective import Objective

class MultiObjective(Objective):
    """
    Class to handle problems with multiple objective functions.

    param func: objective function.
    param n_obj: number of objective functions
    param num_cores: number of cores to use in the process of evaluating the objective (default, 1).
    param objective_name: name of the objective function.
    param batch_type: Type of batch used. Only 'synchronous' evaluations are possible at the moment.
    param space: Not in use.

    """


    def __init__(self, func, n_obj, num_cores = 1, objective_name = 'no_name', batch_type = 'synchronous', space = None):
        self.func  = func
        self.n_procs = num_cores
        self.num_evaluations = 0
        self.space = space
        self.objective_name = objective_name
        self.n_obj = n_obj


    def evaluate(self, x):
        """
        Performs the evaluation of the objective at x.
        """

        f_evals, cost_evals = self._eval_func(x)
        return f_evals, cost_evals


    def _eval_func(self, x):
        """
        Performs sequential evaluations of the function at x (single location or batch). The computing time of each
        evaluation is also provided.
        """
        cost_evals = []
        f_evals     = np.empty(shape=[0, self.n_obj])

        for i in range(x.shape[0]):
            st_time    = time.time()
            rlt = self.func(np.atleast_2d(x[i]))
            f_evals     = np.vstack([f_evals,rlt])
            cost_evals += [time.time()-st_time]
        return f_evals, cost_evals
