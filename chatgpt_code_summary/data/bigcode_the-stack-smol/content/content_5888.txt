"""
pyAutoSpec

Spectral learning for WFA/MPS
"""

from .wfa import Wfa, SpectralLearning
from .mps import Mps
from .plots import parallel_plot
from .function_wfa import FunctionWfa
from .function_mps import FunctionMps
from .dataset_mps import DatasetMps
from .image_wfa import ImageWfa

__all__ = ["Wfa", "Mps", "parallel_plot", "SpectralLearning", "FunctionWfa", "FunctionMps", "DatasetMps", "ImageWfa"]
