## this tool is the core function of cnv and snv analysis
## author: taozhou
## email: zhou.tao@genecast.com.cn

import matplotlib as mpl
mpl.use('Agg')
import warnings
warnings.filterwarnings("ignore")
import itertools
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.colors as mc
from genecast_package.svm_analysis import feature_select, evaluate_model
from sklearn.decomposition import PCA
from collections import OrderedDict
from collections import defaultdict
import datetime
import pandas as pd
from scipy.stats import ranksums
import os
import sh
import warnings
warnings.filterwarnings("ignore")


def z_score(data, axis):
    if axis == 3:
        return data
    if axis == 1:
        z_scored = data
    else:
        z_scored = data.T

    z_scored = (z_scored - z_scored.mean()) / z_scored.std()

    if axis == 1:
        return z_scored
    else:
        return z_scored.T


def pheatmap(data, length, col_cluster=True, xticklabels=True, yticklabels=True, color=None, name=None, args=None):
    data = z_score(data, axis=args.z_score)
    if len(data.columns) > 30:
        xticklabels = False
    if len(data) > 80:
        yticklabels = False
    vmin, vmax = data.unstack().quantile([.05, .95])
    if args.z_score == 3:
        vmin, vmax = 0, 4
    re = sns.clustermap(data, cmap=args.cmp, row_cluster=True, method=args.cluster_method, col_cluster=col_cluster, figsize=(13, 10), \
                        xticklabels=True, yticklabels=yticklabels, vmin=vmin, vmax=vmax, col_colors=color)
    re.ax_heatmap.set_xticklabels(re.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    re.ax_heatmap.set_yticklabels(re.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    if col_cluster == False:
        for group, number in length.items():
            re.ax_col_colors.text((number[0] + number[1])/2 + 1.5 - len(group)/2, 1.2, group, size=30)
        re.savefig(name + "." + args.save)
    else:
        re.savefig(name + "_col_cluster." + args.save)
    plt.close()


def make_col_color_heatmap(group_dic, args=None):
    common_color = ["blue", "red", "green", "grey"]
    color = {}; length = {}
    temp = 0
    i = 0
    for name, group in group_dic.items():
        length[name] = [temp, temp + len(group)]
        temp += len(group)
        for sample in group:
            color[sample] = common_color[i]
        i += 1
    if args.ac and args.bc:
        color[group1] = args.ac
        color[group2] = args.bc
    color = pd.Series(color)
    color.name = "group"
    return color, length


def pca(data, group_dic, n=None, args=None):
    pca = PCA(n_components=2)
    group = []
    length = OrderedDict()
    temp = 0
    for name, g in group_dic.items():
        length[name] = [temp, temp + len(g)]
        temp += len(g)
        group += g
    data = data[group]
    newData = pca.fit_transform(data.T)
    colors = {}
    colors1 = ["blue", "red", "green", 'turquoise', "grey"]
    i = 0
    for name, number in length.items():
        colors[name] = colors1[i]
        i += 1
    if args.ac and args.bc:
        colors[group1] = args.ac
        colors[group2] = args.bc
    for name, number in length.items():
        plt.scatter(newData[number[0]:number[1], 0], newData[number[0]:number[1], 1], label=name, color=colors[name])
    plt.title("PCA analysis", size=20)
    pc1 = 100*pca.explained_variance_ratio_[0]
    pc2 = 100*pca.explained_variance_ratio_[1]
    plt.xlabel("PC1(%.1f)" % pc1, size=15)
    plt.ylabel("PC1(%.1f)" % pc2, size=15)
    plt.legend()
    plt.savefig("PCA_%s.png" % n)
    plt.close()


def plot_box(data, which, outname, palette, regulation, group, args=None):
    fig, ax1 = plt.subplots(figsize=(8,12))
    box_data = defaultdict(list)
    names = []
    if which == "cnv":
        how = "mean"
        for name, g in group.items():
            names.append(name)
            box_data[name] = data[g]
    else:
        how = "sum"
        for name, g in group.items():
            names.append(name)
            box_data[name] = data[g]
    z, p = ranksums(box_data[names[0]], box_data[names[1]])
    if p >= 0.05:
        plt.close()
        return
    data.to_csv(outname + "_box_data_%s" % (regulation) + ".txt", sep="\t")
    if args.ac and args.bc:
        group1 = list(group.keys())[0]
        group2 = list(group.keys())[1]
        palette[group1] = args.ac
        palette[group2] = args.bc
    sns.boxplot(data=pd.DataFrame(box_data), ax=ax1, width=0.2, linewidth=.5, palette=palette)
    ax1.set_title("Difference of %s (p = %f)" % (which, p), size=30)
    ax1.set_ylabel('%s value' % (which), size=30)
    fig.autofmt_xdate(ha='center', rotation=0)
    plt.xticks(rotation=0, size=30)
    plt.legend()
    fig.savefig(r'%s_box_data_%s_%s_Boxplot.%s' % (outname, regulation, how, args.save), dpi=600, size=0.5)
    plt.close()


def databox(raw, which, outname=None, group=None, args=None):
    palette_up = {}; palette_down = {}
    up = []; down = []
    group1_data = raw[list(group.values())[0]]; group1 = list(group.keys())[0]
    group2_data = raw[list(group.values())[1]]; group2 = list(group.keys())[1]
    for gene in raw.index:
        if group1_data.ix[gene].sum() - group2_data.ix[gene].sum() >= 0:
            up.append(gene); palette_up[group1] = "red"; palette_up[group2] = "blue"
        else:
            down.append(gene); palette_down[group1] = "blue"; palette_down[group2] = "red"
    if len(palette_up) > 0:
        for i in up:
            plot_box(raw.ix[i], which, i, palette_up, "up", group, args=args)
    if len(palette_down) > 0:
        for i in down:
            plot_box(raw.ix[i], which, i, palette_down, "down", group, args=args)


def save_data_pdf(data, name, length, color, group_dic, which, args=None):
    data.to_csv("%s.txt" % name, sep="\t")
    length = {key.split("/")[-1]: value for key, value in length.items()}
    group_dic = {key.split("/")[-1]: value for key, value in group_dic.items()}
    try:
        pheatmap(data, length, col_cluster=True, color=color, name=name, args=args)
        pheatmap(data, length, col_cluster=False, color=color, name=name, args=args)
    except MemoryError:
        print("you gene need too much MemoryError and i, so pass and do next")
    pca(data, group_dic, n=name, args=args)
    databox(data, which, outname=name, group=group_dic, args=args)


def save_parameters(args=None):
    f = open("parameters.txt", "w")
    for arg in dir(args):
        if not arg.startswith("_"):
            f.write(arg + ": " + str(getattr(args, arg)) + "\n")
    f.close()


def make_result_folder(args=None, which="cnv", fun=None):
    feature_genes = []; gene_lists = {}; color_length = {}
    os.chdir(args.outdir)
    i = datetime.datetime.now()
    # for two_group in itertools.combinations([args.group1, args.group2], 2):
    two_group = [args.group1[0].split("/")[-2], args.group2[0].split("/")[-2]]
    target = args.group1[0].split("/")[-2] + "_VS_" + args.group2[0].split("/")[-2] + "_%s%s%s_%s%s" % (i.year, i.month, i.day, i.hour, i.minute)
    try:
        os.mkdir(target)
    except FileExistsError:
        sh.rm("-rf",target)
        os.mkdir(target)
    if which == "cnv":
        name = "cnv_median_" + args.data_type
        gene_list, a_group, b_group = fun(args=args)
    else:
        if args.cal_type == "num":
            name = "snv_number"
        else:
            name = "snv_mean"
        gene_list, a_group, b_group = fun(args=args)
    # feature_gene = feature_select(gene_list, a_group, b_group, pval=args.pval, method=args.feature_selection_method,\
                                  # criterion=args.criterion, penalty=args.penalty, C=args.C, threshold=args.threshold)
    feature_gene = feature_select(gene_list, a_group, b_group, args=args)
    feature_genes.append(feature_gene)
    gene_lists[two_group[0]] = gene_list[a_group]; gene_lists[two_group[1]] = gene_list[b_group]
    os.chdir(target)
    save_parameters(args=args)
    group_dic = {two_group[0]: a_group, two_group[1]: b_group}
    color_length[two_group[0]] = a_group; color_length[two_group[1]] = b_group
    color, length = make_col_color_heatmap(group_dic, args=args)
    save_data_pdf(gene_list, "host_gene_%s" % name, length, color, group_dic, which, args=args)
    pd.DataFrame({"gene":feature_gene}).to_csv("feature_gene_pval%0.2f.txt" % args.pval, sep="\t", index=False)
    feature_gene_cnv = gene_list.ix[feature_gene]
    evaluate_model(gene_list, a_group, b_group, feature_gene, name="feature_gene_%s" % name, args=args)
    save_data_pdf(feature_gene_cnv, "feature_gene_%s" % name, length, color, group_dic, which, args=args)
    os.chdir(args.outdir)
    # if len(args.group1 + args.group2) > 2:
        # try:
            # os.mkdir("intersection")
        # except FileExistsError:
            # pass
        # os.chdir("intersection")
        # color, length = make_col_color_heatmap(color_length)
        # intersection_feature_gene = list(set(feature_genes[0]).intersection(*feature_genes[1:]))
        # intersection_feature_gene_cnv = pd.concat([data.ix[intersection_feature_gene] for [args.group1, args.group2], data in gene_lists.items()], axis=1)
        # try:
            # save_data_pdf(intersection_feature_gene_cnv, "intersection", length, color, color_length)
        # except Exception:
            # print("no intersection\njob finish...")
        # os.chdir(args.outdir)
