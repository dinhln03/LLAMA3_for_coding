"""A module for useful functions. 

:author: Matthew Gidden <matthew.gidden _at_ gmail.com>
"""
import numpy as np   

rms = lambda a, axis=None: np.sqrt(np.mean(np.square(a), axis=axis))
