import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

from restools.plotting import rasterise_and_save
from papers.jfm2020_probabilistic_protocol.data import Summary as SummaryProbProto
from papers.jfm2020_probabilistic_protocol.extensions import LaminarisationProbabilityFittingFunction2020JFM
from comsdk.comaux import load_from_json


def plot_pretty_tick(ax, val, on_x=False, on_y=False):
    if on_x:
        ax.plot([val, val], [0.0, 0.025], 'k-', linewidth=2)
    if on_y:
        ax.plot([-0.002, -0.0018], [val, val], 'k-', linewidth=2)


def plot_pretty_annotation_on_axis(ax, val, up_lim, text='', on_x=False, on_y=False, shift=None):
    if on_x:
        ax.plot([val, val], [0.04, up_lim], 'k--', linewidth=1)
        shift = 0.0002 if shift is None else shift
        ax.text(val + shift, 0.03, text, fontsize=14)
    if on_y:
        ax.plot([-0.0016, up_lim], [val, val], 'k--', linewidth=1)
        shift = -0.05 if shift is None else shift
        ax.text(-0.0017, val + shift, text, fontsize=14)
    plot_pretty_tick(ax, val, on_x=on_x, on_y=on_y)


def plot_posterior_distribution(ax, N_lam, N_turb, obj_to_rasterize, simplified=False):
    a = N_lam + 1./2
    b = N_turb + 1./2
    ax.plot(x, beta.pdf(x, a, b), label=r'$N = ' + str(N_lam + N_turb) + '$\n$l = ' + str(N_lam) + r'$',
            linewidth=4 if simplified else 2)
    ax.plot([beta.mean(a, b)], [beta.pdf(beta.mean(a, b), a, b)], 'ro',
            markersize=12 if simplified else 8)
    _, y_max = ax.get_ylim()
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if simplified:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        obj = ax.fill_between([beta.ppf(0.1, a, b), beta.ppf(0.9, a, b)], [0., 0.], [y_max, y_max], alpha=0.3)
        obj_to_rasterize.append(obj)
        ax.legend(fontsize=12)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel(r'$p$')


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    # Plot fitting sketch

    summary_prob_proto = load_from_json(SummaryProbProto)
    energies = 0.5 * np.r_[[0.], summary_prob_proto.energy_levels]
    energies = energies[0:-20]
    print(summary_prob_proto.confs[1].description)
    p_lam = np.r_[[1.], summary_prob_proto.confs[1].p_lam]
    p_lam = p_lam[0:-20]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bar_width = 0.0004
    # plot p_lam bars
    obj = ax.bar(energies, p_lam, 2*bar_width, alpha=0.75, color='lightpink', zorder=0)
    # plot fitting
    fitting = LaminarisationProbabilityFittingFunction2020JFM.from_data(energies, p_lam)
    e = np.linspace(energies[0], energies[-1], 200)
    ax.plot(e, fitting(e), color='blue', zorder=0, linewidth=3)
    E_99 = fitting.energy_with_99_lam_prob()
    E_flex = fitting.energy_at_inflection_point()
    E_a = fitting.energy_close_to_asymptote()
    ax.plot([E_99], [fitting(E_99)], 'ko', markersize=10)
    ax.plot([E_flex], [fitting(E_flex)], 'ko', markersize=10)
    ax.plot([E_a], [fitting(E_a)], 'ko', markersize=10)
    ax.set_xlim((-0.002, energies[-1]))
    ax.set_ylim((0, 1.05))
    energies_ticks = [None for _ in range(len(energies))]
    for i in range(3):
        energies_ticks[i] = r'$E^{(' + str(i + 1) + ')}$'
    energies_ticks[3] = r'...'
    energies_ticks[10] = r'$E^{(j)}$'
#    ax.annotate(r'$\mathbb{E} P_{lam}(E^{(j)})$', xy=(energies[10], p_lam[10]), xytext=(energies[10] - 0.002, p_lam[10] + 0.2),
#                arrowprops=dict(arrowstyle='->'), fontsize=16)
    ax.annotate(r'$\bar{P}_{lam}(E^{(j)})$', xy=(energies[10], p_lam[10]), xytext=(energies[10] - 0.002, p_lam[10] + 0.2),
                arrowprops=dict(arrowstyle='->'), fontsize=16)
    ax.annotate(r'$p(E)$', xy=(energies[4], fitting(energies[4])), xytext=(energies[4] + 0.001, fitting(energies[4]) + 0.2),
                arrowprops=dict(arrowstyle='->'), fontsize=16)

    col_labels = ['Definition']
    row_labels = [
        r'$E_{99\%}$',
        r'$E_{flex}$',
        r'$E_{a}$',
        r'$a$'
    ]
    table_vals = [
        [r'$p(E_{99\%})=0.99$'],
        [r'$E_{flex} = \dfrac{\alpha - 1}{\beta}$'],
        [r'$|p(E_{a}) - a| = 0.01$'],
        [r'$a = \lim_{E \to \infty} p(E)$']
    ]
    # the rectangle is where I want to place the table
    table = plt.table(cellText=table_vals,
                      colWidths=[0.2]*3,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='upper right')
    #table.auto_set_font_size(False)
    #table.set_fontsize(16)
    table.scale(1, 3)
    ax.set_xticks(energies)
    ax.set_xticklabels(energies_ticks)

    plot_pretty_annotation_on_axis(ax, fitting.asymp, energies[-1], text=r'$a$', on_y=True)
    plot_pretty_annotation_on_axis(ax, 0.99, E_99, text=r'$0.99$', on_y=True)
    plot_pretty_annotation_on_axis(ax, E_99, 0.99, text=r'$E_{99\%}$', on_x=True)
    plot_pretty_annotation_on_axis(ax, E_flex, fitting(E_flex), text=r'$E_{flex}$', on_x=True)
    plot_pretty_annotation_on_axis(ax, E_a, fitting(E_a), text=r'$E_{a}$', on_x=True)
#    ax_secondary_yaxis = ax.twinx()
#    ax_secondary_yaxis.tick_params(axis="y", direction="in", pad=-25.)
#    ax_secondary_yaxis.yaxis.tick_left()
#    #ax_secondary_yaxis = ax.secondary_yaxis("left", xticks=[fitting.asymp, 0.99],
#    #                               xticklabels=[r'$a$', 0.99])
#    ax_secondary_yaxis.set_yticks([fitting.asymp, 0.99])
#    ax_secondary_yaxis.set_yticklabels([r'$a$', r'$0.99$'])
#    for ax_ in (ax, ax_secondary_yaxis):
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.tight_layout()
    plt.savefig('fitting_sketch.eps')
    plt.show()

    # Plot posterior distribution examples
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    small_sample_cases = [
        (0, 10),  # first number = number of lam events; second number = number of turb events
        (3, 7),
        (9, 1),
    ]
    large_sample_cases = [
        (0, 30),  # first number = number of lam events; second number = number of turb events
        (9, 21),
        (27, 3),
    ]
    x = np.linspace(0., 1., 200)
    obj_to_rasterize = []
    for ax, (N_lam, N_turb) in zip(axes, small_sample_cases):
        plot_posterior_distribution(ax, N_lam, N_turb, obj_to_rasterize)
    for ax, (N_lam, N_turb) in zip(axes, large_sample_cases):
        plot_posterior_distribution(ax, N_lam, N_turb, obj_to_rasterize)
    for ax in axes:
        ax.grid()
    axes[0].set_ylabel(r'$f_{P_{lam}}(p | \boldsymbol{S} = \boldsymbol{s})$')
    plt.tight_layout()
    fname = 'posterior_examples.eps'
    rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
    plt.show()

    # Plot small posterior distributions for large schema
    sample_cases = [
        (0, 10),  # first number = number of lam events; second number = number of turb events
        (2, 8),
        (7, 3),
    ]
    for N_lam, N_turb in sample_cases:
        fig, ax = plt.subplots(1, 1, figsize=(3, 2.3))
        x = np.linspace(0., 1., 200)
        obj_to_rasterize = []
        plot_posterior_distribution(ax, N_lam, N_turb, obj_to_rasterize, simplified=True)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.tight_layout()
        fname = 'posterior_example_{}_{}.eps'.format(N_lam, N_turb)
        rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
        plt.show()

    # Plot fitting sketch

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    bar_width = 0.0004
    # plot p_lam bars
    obj = ax.bar(energies, p_lam, 2*bar_width, alpha=0.75, color='lightpink', zorder=0)
    # plot fitting
    fitting = LaminarisationProbabilityFittingFunction2020JFM.from_data(energies, p_lam)
    e = np.linspace(energies[0], energies[-1], 200)
    ax.plot(e, fitting(e), color='blue', zorder=0, linewidth=3)
    ax.set_xlim((-0.002, energies[-1]))
    ax.set_ylim((0, 1.05))
    energies_ticks = [None for _ in range(len(energies))]
    for i in range(3):
        energies_ticks[i] = r'$E^{(' + str(i + 1) + ')}$'
    energies_ticks[3] = r'...'
    energies_ticks[10] = r'$E^{(j)}$'
    ax.set_xticks(energies)
    ax.set_xticklabels(energies_ticks)

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.tight_layout()
    plt.savefig('fitting_sketch_simplified.eps')
    plt.show()
