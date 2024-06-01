##################
### original author: Parashar Dhapola
### modified by Rintu Kutum
##################
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re
import json
import pybedtools as pbt
import collections
from scipy.stats import ttest_ind, sem, mannwhitneyu, gaussian_kde, zscore, wilcoxon, norm, poisson
from scipy import ndimage
from scipy.integrate import simps
import os
import glob
import itertools
import pysam
import tables
import sys

closest_bed_dist_jsons = glob.glob(pathname='./data/Histones/dist_json_formatted/*.json')
bed_closest_data = {}
bed_counts = {}
for i in closest_bed_dist_jsons:
	print "\rProcessing\t%s\n" % i,
	mark = i.split('/')[-1].split('_')[0]
	cell = i.split('/')[-1].split('_', 1)[-1].split('.')[0]
	if mark not in bed_counts:
		bed_counts[mark] = {}
	bed_counts[mark][cell] = pbt.BedTool('./data/Histones/bed_formatted/%s_%s.bed' % (mark, cell)).count()
	if mark not in bed_closest_data:
		bed_closest_data[mark] = {}
	bed_closest_data[mark][cell] = json.load(open(i))

active_marks = ['H3k4me1', 'H3k4me2', 'H3k4me3', 'H3k9ac', 'H3k27ac', 'H4k20me1']
repress_marks = ['H3k9me1', 'H3k9me3', 'H3k27me3']
other_marks = ['H2az', 'H3k36me3', 'H3k79me2']
window = 10000
binsize = 200
window_frac_sig = 0.1
mpl.style.use('seaborn-whitegrid')

def get_smoothend_curve(array, smoothen=True, sigma=3, z_norm=False, log2=False):
    a = array.copy()
    if log2 is True:
        a = np.log2(a)
    if z_norm is True:
        a = zscore(a)
    if smoothen is True:
        return ndimage.gaussian_filter1d(a, sigma)
    else:
        return a

def make_stats(t,c,w,b,swp):
    u = int(w/b-w/b*swp)
    d = int(w/b+w/b*swp)
    mu = np.mean([np.mean(i[u:d]) for i in c])
    return {
        'vals': t*10000/bed_counts[mark][cell],
        'shuffle_vals': c[:20]*10000/bed_counts[mark][cell],
        'total_histone_marks': bed_counts[mark][cell],
        'marks_sig_window': np.sum(t[u:d]),
        'marks_full_window': np.sum(t),
        'pval': 1-poisson(mu).cdf(np.mean(t[u:d])),
    }

print "\rGenerating\t%s\n" % 'Figure-3A-3B-3C:',
stats = {}
for mark_set, nc, name in zip([active_marks, repress_marks, other_marks],
                               [2,1,1], ['activation', 'repression', 'others']):
    nr = 3
    fig, ax = plt.subplots(nr, nc, figsize=(1+5*nc, 12))
    row = 0
    col = 0
    for mark in mark_set:
        print (mark)
        all_marks = []
        all_controls = []
        stats[mark] = {}
        for cell in bed_closest_data[mark]:
            t = np.array(bed_closest_data[mark][cell]['closest_dist'])
            c = np.array(bed_closest_data[mark][cell]['shuffle_dist'])
            stats[mark][cell] = make_stats(t, c, window, binsize, window_frac_sig)
        x = np.asarray([i for i in range(len(t))])
        if nc > 1:
            axes = ax[row, col]
        else:
            axes = ax[row]
        for cell in stats[mark]:
            for shuffle in stats[mark][cell]['shuffle_vals']:
                axes.plot(x, get_smoothend_curve(shuffle, z_norm=False, log2=True, smoothen=True),
                          alpha=0.2, c='lightgrey', linewidth=0.5)
        for cell in stats[mark]:
            if stats[mark][cell]['pval'] < 1e-2:
                color = 'crimson'
            else:
                color = 'dimgrey'
            axes.plot(x, get_smoothend_curve(stats[mark][cell]['vals'],
                      z_norm=False, log2=True, smoothen=True), alpha=0.7, c=color, linewidth=1.3)
        axes.set_title(mark, fontsize=24)
        axes.axvline(window/binsize, ls='--')
        axes.axvspan(window/binsize-window/binsize*window_frac_sig,
                             window/binsize+window/binsize*window_frac_sig,
                             alpha=0.2, color='dodgerblue')
        axes.set_xticks(list(map(int, np.linspace(0,(2*window)/binsize,9))))
        axes.set_xlim((0,(2*window)/binsize))
        _ = [tick.label.set_fontsize(20) for tick in axes.yaxis.get_major_ticks()]
        if col == 0:
            #axes.set_ylabel('Log2 (histone\nmarks per 10K\nmarks in sample)', fontsize=22)
            axes.set_ylabel('Log2 (normalized\nhistone peaks)', fontsize=22)
        if row == nr-1:
            axes.set_xlabel('Distance from TRF2 peak center', fontsize=22)
            axes.set_xticklabels(map(int, np.linspace(-window,window,9)), fontsize=20, rotation=45)
        else:
            axes.set_xticklabels([])
        col+=1
        if col == nc:
            col = 0
            row+=1
    fig.tight_layout()
    fig.savefig('./figures/Figure-3_histone_%s.png' % name, dpi=300)

for i in stats:
    n = 0
    ns = 0
    for j in stats[i]:
        n+=1
        if stats[i][j]['pval'] < 0.01:
            ns+=1
    print (i, n, ns)

print "\rGenerating\t%s\n" % 'Figure-4A:',
import seaborn as sns
count_histone_df = []
for mark in stats:
    for cell in stats[mark]:
        count_histone_df.append([mark, cell,
            stats[mark][cell]['total_histone_marks']])
count_histone_df = pd.DataFrame(count_histone_df, columns=['Mark', 'Cell', 'Value'])
fig, ax = plt.subplots(1,1, figsize=(14,5))
sns.set_style("whitegrid")
sns.violinplot(x="Mark", y="Value", data=count_histone_df, ax=ax, inner='point', c='Grey', saturation=0,
               scale="width", order=active_marks+repress_marks+other_marks, scale_hue=True)
_ = ax.set_xticklabels(active_marks+repress_marks+other_marks, rotation=70, fontsize=24)
ax.set_title('Distribution of nubmer of histone peaks in cell lines for each histone mark', fontsize=26)
ax.set_xlabel('')
ax.set_ylabel('Number of histone peaks', fontsize=24)
_ = [tick.label.set_fontsize(24) for tick in ax.yaxis.get_major_ticks()]
sns.despine()
fig.tight_layout()
fig.savefig('./figures/Suppl-Figure-4A-histone-dist.png', dpi=300)

print "\rGenerating\t%s\n" % 'Figure-4B:',
# TRF2 +/-10KB
count_histone_df = []
for mark in stats:
    for cell in stats[mark]:
        count_histone_df.append([mark, cell,
            stats[mark][cell]['marks_full_window']*10000/stats[mark][cell]['total_histone_marks']])
count_histone_df = pd.DataFrame(count_histone_df, columns=['Mark', 'Cell', 'Value'])
fig, ax = plt.subplots(1,1, figsize=(14,5))
sns.set_style("whitegrid")
sns.violinplot(x="Mark", y="Value", data=count_histone_df, ax=ax, inner='point', c='Grey', saturation=0,
               scale="width", order=active_marks+repress_marks+other_marks, scale_hue=True)
_ = ax.set_xticklabels(active_marks+repress_marks+other_marks, rotation=70, fontsize=24)
ax.set_title('Distribution of histone peaks in +/- 10KB of TRF2 peaks', fontsize=26)
ax.set_xlabel('')
ax.set_ylabel('Number of normalized\nhistone peaks', fontsize=24)
_ = [tick.label.set_fontsize(24) for tick in ax.yaxis.get_major_ticks()]
sns.despine()
fig.tight_layout()
fig.savefig('./figures/Suppl-Figure-4B-histone-peaks-10kb-dist.png', dpi=300)

print "\rGenerating\t%s\n" % 'Figure-4C:',
# TRF2 +/-500bp
count_histone_df = []
for mark in stats:
    for cell in stats[mark]:
        count_histone_df.append([mark, cell,
            stats[mark][cell]['marks_sig_window']*10000/stats[mark][cell]['total_histone_marks']])
count_histone_df = pd.DataFrame(count_histone_df, columns=['Mark', 'Cell', 'Value'])
fig, ax = plt.subplots(1,1, figsize=(14,5))
sns.set_style("whitegrid")
sns.violinplot(x="Mark", y="Value", data=count_histone_df, ax=ax, inner='point', c='Grey', saturation=0,
               scale="width", order=active_marks+repress_marks+other_marks, scale_hue=True)
_ = ax.set_xticklabels(active_marks+repress_marks+other_marks, rotation=70, fontsize=24)
ax.set_title('Distribution of histone peaks in +/- 500bp of TRF2 peaks', fontsize=26)
ax.set_xlabel('')
ax.set_ylabel('Number of normalized\nhistone peaks', fontsize=24)
_ = [tick.label.set_fontsize(24) for tick in ax.yaxis.get_major_ticks()]
sns.despine()
fig.tight_layout()
fig.savefig('./figures/Suppl-Figure-4C-histone-peaks-500bp-dist.png', dpi=300)


print "\rGenerating\t%s\n" % 'Figure-4D:',
# p-values
count_histone_df = []
for mark in stats:
    for cell in stats[mark]:
        count_histone_df.append([mark, cell,
            -np.log10(stats[mark][cell]['pval'])])
count_histone_df = pd.DataFrame(count_histone_df, columns=['Mark', 'Cell', 'Value'])
fig, ax = plt.subplots(1,1, figsize=(14,5))
sns.set_style("whitegrid")
sns.violinplot(x="Mark", y="Value", data=count_histone_df, ax=ax, inner='point', c='Grey', saturation=0,
               scale="width", order=active_marks+repress_marks+other_marks, scale_hue=True)
_ = ax.set_xticklabels(active_marks+repress_marks+other_marks, rotation=70, fontsize=24)
ax.set_title('Distribution of p-values in cell lines', fontsize=26)
ax.set_xlabel('')
ax.set_ylabel('-log10(p-value)', fontsize=24)
_ = [tick.label.set_fontsize(24) for tick in ax.yaxis.get_major_ticks()]
sns.despine()
fig.tight_layout()
fig.savefig('./figures/4D-histone-pval-10kb-dist.png', dpi=300)
