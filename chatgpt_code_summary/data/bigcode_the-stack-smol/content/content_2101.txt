from __future__ import division
import numpy as np
import sys
import os
import shutil
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import argparse
import paraview.simple as parasim
import multiprocessing as mp
import copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'model')))
from geometry_generation import *
matplotlib.rcParams['font.size'] = 6
import scipy.interpolate as interpsci
import seaborn as sns

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def makeImage(snapshots, geometryX, geometryY, data_names, folder_name, times):

    fig = plt.figure(figsize=(7,4.72441/4*3))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(234)
    ax3 = fig.add_subplot(232)
    ax4 = fig.add_subplot(235)
    ax5 = fig.add_subplot(233)
    ax6 = fig.add_subplot(236)
    axes = [[ax1, ax2], [ax3, ax4], [ax5, ax6]]
    all_axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    for k in range(len(snapshots[0])):
        axes_current = axes[k]
        values = [[],[]]
        for j in range(len(data_names)):
            i = snapshots[j][k]
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(data_names[j] + '{0:06d}.vtu'.format(i))
            reader.Update()
            data = reader.GetOutput()
            points = data.GetPoints()
            npts = points.GetNumberOfPoints()
            x = vtk_to_numpy(points.GetData())[:, 0]
            y = vtk_to_numpy(points.GetData())[:, 1]
            f = vtk_to_numpy(data.GetPointData().GetArray(0))
            triangles = vtk_to_numpy(data.GetCells().GetData())
            ntri = triangles.size//4
            tri = np.take(triangles,[m for m in range(triangles.size) if m%4 != 0]).reshape(ntri, 3)

            waterX = np.linspace(0, 60000, 100)
            waterY = np.zeros(100)
            values[j].append(x)
            values[j].append(y)
            values[j].append(tri)
            values[j].append(f)

        levels = np.linspace(0, 1.0, 100, endpoint=True)

        cmap_new = truncate_colormap(plt.get_cmap("BuPu"), 0.25, 1.0)

        maxL = 51100
        bed_interpolator = interpsci.interp1d(geometryX, geometryY, fill_value='extrapolate')
        geometryX = np.linspace(0, 60000, 1000)
        geometryY = bed_interpolator(geometryX)
        axes_current[0].fill_between(waterX, -200, 0, color='#94aec4ff', zorder=-21)
        axes_current[1].fill_between(waterX, -200, 0, color='#94aec4ff', zorder=-21)
        axes_current[0].fill_between(geometryX, -200, geometryY, color='#c69d6eff', zorder=-18)
        axes_current[1].fill_between(geometryX, -200, geometryY, color='#c69d6eff', zorder=-18)

        cnt1 = axes_current[0].tricontourf(values[0][0], values[0][1], values[0][2], values[0][3]*100, 100, cmap=cmap_new, levels=levels, extend='both', zorder=-20)
        cnt2 = axes_current[1].tricontourf(values[1][0], values[1][1], values[1][2], values[1][3]*100, 100, cmap=cmap_new, levels=levels, extend='both', zorder=-20)

        for cnt in [cnt1, cnt2]:
            for c in cnt.collections:
                c.set_edgecolor("face")

        axes_current[0].set_title("t = %.1f years" % (times[k]-0.5))

        print("Processed file number " + str(i) + ".")

    labels = ['a', 'd', 'b', 'e', 'c', 'f']

    for ax in all_axes:
        ax.set_xlim([49400,maxL])
        ax.set_ylim([-200,100])
        ax.set_rasterization_zorder(-10)

    for j in range(len(all_axes)):
        all_axes[j].text(0.025, 0.97, labels[j], transform=all_axes[j].transAxes, va='top', fontsize=8, weight='bold')

    for ax in [ax3, ax4, ax5, ax6]:
        plt.sca(ax)
        ylims = plt.yticks()
        print(ylims)
        locs = ylims[0][1:-1]
        labels = []
        for j in range(len(locs)):
            labels.append('%.2f'%(locs[j]))
        plt.sca(ax)
        plt.yticks(locs, [" "]*len(locs))
    for ax in [ax1, ax3, ax5]:
        plt.sca(ax)
        xlims = plt.xticks()
        print(xlims)
        locs = xlims[0][1:-1]
        labels = []
        for j in range(len(locs)):
            labels.append('%.2f'%(locs[j]))
        plt.sca(ax)
        plt.xticks(locs, [" "]*len(locs))

    for ax in [ax2, ax4, ax6]:
        plt.sca(ax)
        labelsx = [num/1000 for num in locs]
        plt.xticks(locs, labelsx)

    for ax in [ax2, ax4, ax6]:
        ax.set_xlabel('Distance (km)')
    for ax in [ax1, ax2]:
        ax.set_ylabel('Height (m)')
    ax1.text(-0.5, 0.5, 'No Advection', transform=ax1.transAxes, va='center', fontsize=12, rotation='vertical')
    ax2.text(-0.5, 0.5, 'Advection', transform=ax2.transAxes, va='center', fontsize=12, rotation='vertical')

    plt.tight_layout(pad=1.0,h_pad=-1.0,w_pad=0.0)
    fig.savefig(folder_name + "/" + "thumbnails_warm.eps", transparent=False)
    plt.close(fig)


if __name__ == "__main__":
    sns.set(palette='colorblind')
    sns.set(font_scale=0.8)
    sns.set_style(style='ticks')

    starting_directory = os.getcwd()
    os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests')))

    main_directory = os.getcwd()

    geometryX, geometryY, xz_boundary = make_geometry_grounded(-0.01, 50000, -150, 50, 100, 10)
    times = [0.5, 8.0, 15.5]
    directories = ['warm_noad', 'warm']
    dataName = 'width_timeseries'

    data_names = [os.path.join(directory, dataName) for directory in directories]

    snapshots = [[], []]
    for i in range(len(directories)):
        for j in range(len(times)):
            if times[j] == int(0):
                snapshots[i].append(int(0))
            else:
                os.chdir(directories[i])
                reader_paraview = parasim.PVDReader(FileName=dataName + '.pvd')
                times_imported = reader_paraview.GetPropertyValue('TimestepValues')
                times_temp = 0.0
                for k in range(len(times_imported)):
                    if times_imported[k] >= times[j] and times_temp <= times[j]:
                        snapshots[i].append(int(k))
                        break
                    else:
                        times_temp = times_imported[k]
                os.chdir(main_directory)

    os.chdir(main_directory)

    print(snapshots)

    makeImage(snapshots, geometryX, geometryY, data_names, starting_directory, times)