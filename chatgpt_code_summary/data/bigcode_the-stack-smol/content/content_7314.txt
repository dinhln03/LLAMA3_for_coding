#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:28:54 2018

@author: galengao

This is the original analysis code as it exists in the environment where it was writen and initially run.
Portions and modifications of this script constitute all other .py scripts in this directory.
"""
import numpy as np
import pandas as pd

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns


### Helper Function to Load in the Data ###
def load_data(coh, thresh=False):
    """Load in the hg38 and hg19 gistic thresholded data. Assume GISTIC runs 
    for each tumor type live in a parent directory (hg38_gistic or hg19_gistic)
    one level up from this script."""
    if thresh:
        hg38 = '../hg38_gistic/'+coh+'/all_thresholded.by_genes.txt'
        hg19 = '../hg19_gistic/'+coh+'/all_thresholded.by_genes.txt'
        hg38drops = ['Cytoband', 'Locus ID']
    else:
        hg38 = '../hg38_gistic/'+coh+'/all_data_by_genes.txt'
        hg19 = '../hg19_gistic/'+coh+'/all_data_by_genes.txt'
        hg38drops = ['Cytoband', 'Gene ID']
        
    df_hg19 = pd.read_table(hg19, index_col=[0]).drop(['Cytoband', 'Locus ID'], axis=1)
    df_hg38 = pd.read_table(hg38, index_col=[0]).drop(hg38drops, axis=1)
    
    same_samps = list(set(df_hg38.columns) & set(df_hg19.columns))
    same_genes = list(set(df_hg38.index) & set(df_hg19.index))
    print(coh, len(same_genes), len(same_samps))
    return df_hg38[same_samps].T[same_genes], df_hg19[same_samps].T[same_genes]
    
    return df_hg38, df_hg19


### Raw Copy Number Values Analysis Code ###
def raw_value_comparison(coh, plot=False):
    """Return the average differences in raw copy number values between the
    gene-level calls in hg19 and hg38 for each gene for a given tumor type 
    'coh.' If plot=True, plot the genes' differences in a histogram."""
    
    # load in the data
    df_38, df_19 = load_data(coh, thresh=False)

    # compute average sample-by-sample differences for each gene
    df_s = df_38 - df_19
    avg_diff = {g:np.average(df_s[g]) for g in df_s.columns.get_level_values('Gene Symbol')}
    
    # take note of which genes are altered more than our threshold of 4*std
    results = []
    std = np.std([avg_diff[x] for x in avg_diff])
    for g in avg_diff:
        if avg_diff[g] > 4 * std:
            results.append([coh, 'Pos', g, avg_diff[g]])
        elif avg_diff[g] < -4 * std:
            results.append([coh, 'Neg', g, avg_diff[g]])
    
    if plot:
        plt.hist([avg_diff[x] for x in avg_diff], bins=1000)
        plt.title(coh, fontsize=16)
        plt.xlabel('Average CN Difference Between Alignments', fontsize=14)
        plt.ylabel('Genes', fontsize=14)
        sns.despine()
        plt.savefig('./genehists/'+coh+'_genehist.pdf')
        plt.savefig('./genehists/'+coh+'_genehist.png')
        plt.clf()
    
    return results

def sequential_cohort_test_raw_values(cohs, plot=False):
    """Sequentially compare raw gene-level calls for the given tumor types."""
    c_results = []
    for coh in cohs: # perform raw value comparison for each cohort
        c_results += raw_value_comparison(coh, plot=plot)
    
    # compile results together
    df_r = pd.DataFrame(c_results, columns=['Cohort', 'Direction', 'Gene', 'Difference'])
    gcount = Counter(df_r['Gene'])    
    pos_gcount = Counter(df_r[df_r['Direction']=='Pos']['Gene'])
    neg_gcount = Counter(df_r[df_r['Direction']=='Neg']['Gene'])
    df = pd.DataFrame([gcount[x] for x in gcount], index=gcount.keys(), columns=['Count'])
    df['Count_pos'] = [pos_gcount[x] if x in pos_gcount else 0 for x in gcount]
    df['Count_neg'] = [neg_gcount[x] if x in neg_gcount else 0 for x in gcount]

    if plot: # write output
        plt.plot(np.sort([gcount[x] for x in gcount])[::-1], 'b-')
        plt.xlabel('Gene by Rank', fontsize=16)
        plt.ylabel('Number of Occurences', fontsize=16)
        sns.despine()
        plt.savefig('GeneDevianceDropoff.pdf')
        plt.savefig('GeneDevianceDropoff.png')
        df_r.to_csv('./genehists/LargestDifferences.tsv', sep='\t', index=False)
        df.to_csv('./genehists/LargestDifferenceGenes_ByCount.tsv', sep='\t', index=True)


### Thresholded Copy Number Values Analysis Code ###
def thresholded_value_comparison(df_hg38, df_hg19, metric='hamming'):
    """Compare -2,-1,0,1,2 gene-level thresholded calls. metric can be either
    hamming (number of discrepancies in each gene) or manhattan (sum of 
    'distances' between each gene so a 1 to -1 change is 2). Returns a vector
    of each gene's metric."""
    out = []
    for i, g in enumerate(df_hg38.columns):
        if metric == 'hamming':
            out.append(sum(df_hg19[g] != df_hg38[g])/len(df_hg19))
        elif metric == 'manhattan':
            out.append(sum(abs((df_hg19[g] - df_hg38[g]))))
    return pd.DataFrame(out, index=df_hg38.columns)

def sequential_cohort_test_thresholded_values(cohs):
    """Compare thresholded gene-level calls for input tumor types."""
    df_out = pd.DataFrame([])
    for coh in cohs:
        df_hg38, df_hg19 = load_data(coh, thresh=True)
        df_results = thresholded_value_comparison(df_hg38, df_hg19, metric='hamming')
        df_results.columns = [coh]
        df_out = df_out.join(df_results, how='outer')
    
    df_out.to_csv('../readout/DiscordantSampleFractions_perGene_perCohort_thresholdedCalls.tsv', sep='\t')
    return df_out

def plot_fractionDisagreements_perCohort(cohs):
    """Visualize fraction of samples with disagreements in thresholded copy 
    number for each gene. Run sequential_cohort_test_thresholded_values()
    before this function."""
    # Read in data written by sequential_cohort_test_thresholded_values
    df = sequential_cohort_test_thresholded_values(cohs)
    df_box = pd.melt(df.reset_index(), id_vars='Gene Symbol').set_index('Gene Symbol')
    df_box.columns = ['Tumor Type', 'Fraction of Samples with Disagreements']
    dft = df.T
    dft['med_degenerates'] = df.median(axis=0)
    boxorder = dft.sort_values('med_degenerates', axis=0).index

    # read in copy number burden data (requires aneuploidy RecurrentSCNA calls)
    df_cn = pd.read_table('../../PanCanAneuploidy/bin/PANCAN_armonly_ASandpuritycalls_092817_xcellcalls.txt', index_col=0, usecols=[0,1,2,16])
    coh_medians = [int(np.median(df_cn[df_cn['Type']==x]['RecurrentSCNA'].dropna())) for x in df_cn.Type.unique()]
    df_med = pd.DataFrame(coh_medians, index=df_cn.Type.unique(), columns=['med'])

    # plot it out
    pal = sns.color_palette('Blues', max(df_med.med)-min(df_med.med)+1)
    my_pal = {c: pal[df_med.at[c,'med']] for c in df_med.index}
    g = sns.boxplot(x=df_box.columns[0], y=df_box.columns[1], data=df_box, \
                    order=boxorder, fliersize=1, palette=my_pal, linewidth=0.5)
    newxticks = [x+' ('+str(df_med.loc[x]['med'])+')' for x in boxorder]
    g.set_xticklabels(newxticks, rotation=90)
    plt.ylabel('Fraction with Disagreements', fontsize=12)
    sns.despine()
    plt.gcf().set_size_inches((8,3))
    plt.savefig('2_thresholdedCN_boxplot.pdf', bbox_inches='tight')
    plt.savefig('2_thresholdedCN_boxplot.png', bbox_inches='tight')


### Significantly Altered Focal Peaks Analysis Code ###
def peakgene_overlaps(combos, same_genes, normalize=False):
    """Count the number of genes that overlap when examing the hg19 & hg38 
    GISTIC runs' focal peaks."""
    venn_numbers, gsu, gsi = [], [], []
    for coh, ad in combos:
        print(coh)
        # put all significant genes in a list
        fnames = ['../hg19_gistic/'+coh+ad+'genes.conf_99.txt', '../hg38_gistic/'+coh+ad+'genes.txt']
        df38 = pd.read_table(fnames[0], index_col=0).drop(['q value','residual q value','wide peak boundaries'])
        df19 = pd.read_table(fnames[1], index_col=0).drop(['q value','residual q value','wide peak boundaries'])
        g_38 = set([x for col in df38.columns for x in df38[col].dropna()]) & same_genes
        g_19 = set([x for col in df19.columns for x in df19[col].dropna()]) & same_genes
        intersect, union = g_38 & g_19, g_38 | g_19
        gsu.append(union)
        gsi.append(intersect)
        if normalize:
            venn_numbers.append([len(g_19-intersect)/len(union),len(intersect)/len(union), len(g_38-intersect)/len(union)])
        else:
            venn_numbers.append([len(g_19-intersect),len(intersect), len(g_38-intersect)])

    index = [x[0]+'_'+x[1][1:-1] for x in combos]
    return pd.DataFrame(venn_numbers, index=index, columns=['hg19 only','Intersection','hg38 only'])

def plot_peakgene_overlaps(combos, same_genes, write=False):
    """Visualize the results of peakgene_overlaps function in bargraph form."""
    df_out = peakgene_overlaps(combos, same_genes, normalize=False)
    df_d, df_a = df_out[df_out.index.str.split('_').str[-1] == 'del'], \
                    df_out[df_out.index.str.split('_').str[-1] == 'amp']
    for x in zip((df_d, df_a), ('Deletion Peak Memberships', 'Amplification Peak Memberships')):
        x[0].index = x[0].index.str.split('_').str[0]
        x[0].plot.bar(stacked=True, color=['#af8dc3', '#f7f7f7', '#7fbf7b'], linewidth=1, edgecolor='k')
        plt.gca().set_xticklabels(x[0].index, rotation=90)
        plt.title(x[1], fontsize=18)
        plt.gcf().set_size_inches(10,8)
        sns.despine()
        plt.savefig(x[1].split(' ')[0]+'_peakMemberships.pdf', bbox_inches='tight')
        plt.savefig(x[1].split(' ')[0]+'_peakMemberships.png', bbox_inches='tight')
        plt.clf()
    if write:
        df_out.to_csv('VennStats_focalpeaks.tsv', sep='\t')


### Conservation of Significant Copy Number Driver Events Analysis Code ###
def documented_driver_differences():
    """Scan and analyze manually currated DocumentedDriverDifferences.txt file.
    Returns: 1) Number of driver genes called in both hg19 & hg38 GISTIC peaks
    2) Number of drivers missing in hg38 peaks that appeared in hg19 peaks  and
    3) Number of drivers present in hg38 peaks but absent from hg19 peaks."""
    # read in table of documented driver differences
    # (this table needs a manual curation to be generated)
    df = pd.read_table('../DocumentedDriverDifferences.txt', index_col=0)
    # process entries to have just yes/no calls (without parens & brackets)
    df['hg19?'] = df['present in hg19?'].str.strip(')').str.strip('(').str.strip('[').str.strip(']')
    df['hg38?'] = df['present in hg38?'].str.strip(')').str.strip('(').str.strip('[').str.strip(']')

    # number of documented drivers that match in hg19 & hg38
    matches = sum(df['hg19?'] == df['hg38?'])
    # number of documented drivers that are in hg19 but not hg38 & vice versa
    lostdrivers = len(df[(df['hg19?'] == 'yes') & (df['hg38?'] == 'no')])
    recovereddrivers = len(df[(df['hg19?'] == 'no') & (df['hg38?'] == 'yes')])
    
    # Return in order
    return matches, lostdrivers, recovereddrivers



# set up the tumor types we want to analyze
cohs = ['ACC','BLCA','CESC','CHOL','COAD','DLBC','ESCA','GBM', 'HNSC','KICH',\
        'KIRC','KIRP','LAML','LGG','LIHC','LUAD','LUSC','OV','PAAD','PCPG',\
        'PRAD','READ','SARC','SKCM','STAD','TGCT','THCA','THYM','UCEC','UCS','UVM']    
ads = ['/amp_', '/del_']
combos = [(c, a) for c in cohs for a in ads]

# grab list of genes present in both hg19 & hg38
df_hg38 = pd.read_table('../hg38_gistic/CHOL/all_thresholded.by_genes.txt', index_col=0, usecols=[0,1])
df_hg19 = pd.read_table('../hg19_gistic/CHOL/all_thresholded.by_genes.txt', index_col=0, usecols=[0,1])
same_genes = set(df_hg38.index) & set(df_hg19.index)


# action lines -- run the analysis
sequential_cohort_test_raw_values(cohs, plot=True)
plot_fractionDisagreements_perCohort(cohs)
plot_peakgene_overlaps(combos, same_genes, write=True)
print(documented_driver_differences())
