#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Twinboundary plot

This module provide various kinds of plot related to twin boudnary.
"""

import numpy as np
from copy import deepcopy
from twinpy.plot.base import line_chart


def plot_plane(ax,
               distances:list,
               z_coords:list,
               label:str=None,
               decorate:bool=True,
               show_half:bool=False,
               **kwargs):
    """
    Plot plane.

    Args:
        ax: matplotlib ax.
        distances (list): List of plane intervals.
        z_coords (list): List of z coordinate of each plane.
        label (str): Plot label.
        decorate (bool): If True, ax is decorated.
        show_half: If True, atom planes which are periodically equivalent are
                   not showed.
    """
    if decorate:
        xlabel = 'Distance'
        ylabel = 'Hight'
    else:
        xlabel = ylabel = None

    _distances = deepcopy(distances)
    _z_coords = deepcopy(z_coords)
    _distances.insert(0, distances[-1])
    _distances.append(distances[0])
    _z_coords.insert(0, -distances[-1])
    _z_coords.append(z_coords[-1]+distances[0])

    c = np.sum(distances)
    fixed_z_coords = _z_coords + distances[0] / 2 - c / 2
    num = len(fixed_z_coords)
    bulk_distance = _distances[int(num/4)]

    if show_half:
        n = int((num + 2) / 4)
        _distances = _distances[n:3*n]
        fixed_z_coords = fixed_z_coords[n:3*n]
    line_chart(ax=ax,
               xdata=_distances,
               ydata=fixed_z_coords,
               xlabel=xlabel,
               ylabel=ylabel,
               label=label,
               sort_by='y',
               **kwargs)

    if decorate:
        xmin = bulk_distance - 0.025
        xmax = bulk_distance + 0.025
        if show_half:
            ax.hlines(0,
                      xmin=xmin-0.01,
                      xmax=xmax+0.01,
                      linestyle='--',
                      color='k',
                      linewidth=1.)
        else:
            tb_idx = [1, int(num/2), num-1]
            for idx in tb_idx:
                ax.hlines(fixed_z_coords[idx]-distances[0]/2,
                          xmin=xmin-0.01,
                          xmax=xmax+0.01,
                          linestyle='--',
                          color='k',
                          linewidth=1.)


def plot_angle(ax,
               angles:list,
               z_coords:list,
               label:str=None,
               decorate:bool=True):
    """
    Plot angle.

    Args:
        ax: matplotlib ax.
        z_coords (list): List of z coordinate of each plane.
        label (str): Plot label.
        decorate (bool): If True, ax is decorated.
    """
    if decorate:
        xlabel = 'Angle'
        ylabel = 'Hight'
    else:
        xlabel = ylabel = None

    _angles = deepcopy(angles)
    _z_coords = deepcopy(z_coords)
    _angles.append(angles[0])
    _z_coords.append(z_coords[-1]+z_coords[1])

    line_chart(ax=ax,
               xdata=_angles,
               ydata=_z_coords,
               xlabel=xlabel,
               ylabel=ylabel,
               label=label,
               sort_by='y')

    if decorate:
        num = len(_z_coords)
        tb_idx = [0, int(num/2), num-1]
        bulk_angle = angles[int(num/4)]
        for idx in tb_idx:
            ax.hlines(_z_coords[idx],
                      xmin=-1,
                      xmax=bulk_angle+2,
                      linestyle='--',
                      linewidth=1.5)


def plot_pair_distance(ax,
                       pair_distances:list,
                       z_coords:list,
                       label:str=None,
                       decorate:bool=True):
    """
    Plot angle.

    Args:
        ax: matplotlib ax.
        pair_distances (list): List of A-B pair distances, which is originally
                               primitive pair in HCP structure.
        z_coords (list): List of z coordinate of each plane.
        label (str): Plot label.
        decorate (bool): If True, ax is decorated.
    """
    if decorate:
        xlabel = 'Pair Distance'
        ylabel = 'Hight'
    else:
        xlabel = ylabel = None

    _pair_distances = deepcopy(pair_distances)
    _z_coords = deepcopy(z_coords)
    _pair_distances.append(pair_distances[0])
    _z_coords.append(z_coords[-1]+z_coords[1])

    line_chart(ax=ax,
               xdata=_pair_distances,
               ydata=_z_coords,
               xlabel=xlabel,
               ylabel=ylabel,
               label=label,
               sort_by='y')

    if decorate:
        num = len(_z_coords)
        tb_idx = [0, int(num/2), num-1]
        bulk_pair_distance = pair_distances[int(num/4)]
        for idx in tb_idx:
            ax.hlines(_z_coords[idx],
                      xmin=-1,
                      xmax=bulk_pair_distance+2,
                      linestyle='--',
                      linewidth=1.5)
