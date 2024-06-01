# -*- coding: utf-8 -*-
"""
:mod:`plots` -- Tests data plots
================================

.. module:: plots
    :platform: Unix, Windows
    :synopsis: Tests of the raster plots and processed data plots.
.. moduleauthor:: Andre Rocha <rocha.matcomp@gmail.com>
"""

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from src.rocha import plots

@image_comparison(baseline_images=['test_plot'],
                  extensions=['png'])
def test_plot():
    """
    Test the rasters plot as multiples subplots.
    """
    rasters = ['data/relatives/forest_111.tif',
               'data/relatives/forest_112.tif',
               'data/relatives/forest_113.tif',
               'data/relatives/forest_121.tif',
               'data/relatives/forest_122.tif',
               'data/relatives/forest_123.tif',
               'data/relatives/forest_211.tif',
               'data/relatives/forest_212.tif',
               'data/relatives/forest_213.tif',
               'data/relatives/forest_221.tif',
               'data/relatives/forest_222.tif',
               'data/relatives/forest_223.tif']

    title = 'Mean precipitation (mm/day)'
    subtitles = ['HadGEM2 RCP4.5', 'HadGEM2 RCP8.5', 'MIROC5 RCP4.5', 'MIROC5 RCP8.5']
    labels = ['2011-2040', '2041-2070', '2071-2100']

    color = 'RdYlBu_r'

    rows = 3
    cols = 4

    plots.maps(rasters, rows, cols, color, title, subtitles, labels)