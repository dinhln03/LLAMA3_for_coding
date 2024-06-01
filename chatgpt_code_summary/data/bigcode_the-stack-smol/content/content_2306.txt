#!/usr/bin/python

"""Plot LFEs of given order parameter."""

import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from matplotlibstyles import styles
from matplotlibstyles import plotutils


def main():
    args = parse_args()
    f = setup_figure()
    gs = gridspec.GridSpec(1, 1, f)
    ax = f.add_subplot(gs[0, 0])
    if args.post_lfes == None:
        args.post_lfes = ['' for i in range(len(args.systems))]

    plot_figure(f, ax, vars(args))
    setup_axis(ax, args.tag)
    #set_labels(ax)
    save_figure(f, args.plot_filebase)


def setup_figure():
    styles.set_default_style()
    figsize = (plotutils.cm_to_inches(10), plotutils.cm_to_inches(7))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(f, ax, args):
    systems = args['systems']
    varis = args['varis']
    input_dir = args['input_dir']
    tag = args['tag']
    post_lfes = args['post_lfes']
    stacking_enes = args['stacking_enes']

    if stacking_enes is not None:
        stacking_enes = [abs(e) for e in stacking_enes]
        cmap = plotutils.create_truncated_colormap(
            0.2, 0.8, name='plasma')
        #mappable = plotutils.create_linear_mappable(
        #    cmap, abs(stacking_enes[0]), abs(stacking_enes[-1]))
        #colors = [mappable.to_rgba(abs(e)) for e in stacking_enes]
        increment = stacking_enes[1] - stacking_enes[0]
        cmap, norm, colors = plotutils.create_segmented_colormap(cmap, stacking_enes, increment)
    else:
        cmap = cm.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(systems))]

    for i in range(len(systems)):
        system = systems[i]
        vari = varis[i]
        post_lfe = post_lfes[i]
        if post_lfe != '':
            post_lfe = '-' + post_lfe

        inp_filebase = f'{input_dir}/{system}-{vari}_lfes{post_lfe}-{tag}'
        lfes = pd.read_csv(f'{inp_filebase}.aves', sep=' ', index_col=0)
        lfe_stds = pd.read_csv(f'{inp_filebase}.stds', sep=' ', index_col=0)
        temp = lfes.columns[0]
        lfes = lfes[temp]
        lfes = lfes - lfes[0]
        lfe_stds = lfe_stds[temp]

        label = f'{system}-{vari}'
        ax.errorbar(lfes.index, lfes, yerr=lfe_stds, marker='o', label=label,
                    color=colors[i])

    if stacking_enes is not None:
        label = r'$-U_\text{stack} / \SI{1000}{\kb\kelvin}$'
        tick_labels = [f'${e/1000:.1f}$' for e in stacking_enes]
        plotutils.plot_segmented_colorbar(
            f, ax, cmap, norm, label, tick_labels, 'horizontal')


def setup_axis(ax, ylabel=None, xlabel=None, ylim_top=None, xlim_right=None):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(top=ylim_top)
    ax.set_xlim(right=xlim_right)


def set_labels(ax):
    plt.legend()


def save_figure(f, plot_filebase):
    #f.savefig(plot_filebase + '.pgf', transparent=True)
    f.savefig(plot_filebase + '.pdf', transparent=True)
    f.savefig(plot_filebase + '.png', transparent=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory')
    parser.add_argument(
        'plot_filebase',
        type=str,
        help='Plots directory')
    parser.add_argument(
        'tag',
        type=str,
        help='OP tag')
    parser.add_argument(
        '--systems',
        nargs='+',
        type=str,
        help='Systems')
    parser.add_argument(
        '--varis',
        nargs='+',
        type=str,
        help='Simulation variants')
    parser.add_argument(
        '--post_lfes',
        nargs='+',
        type=str,
        help='Filename additions after lfes, if any')
    parser.add_argument(
        '--stacking_enes',
        nargs='+',
        type=float,
        help='Stacking energies (for colormap)')

    return parser.parse_args()


if __name__ == '__main__':
    main()
