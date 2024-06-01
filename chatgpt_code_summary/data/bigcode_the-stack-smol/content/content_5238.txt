# coding: utf-8

import warnings
import numpy as np
import pandas as pd
from packaging import version
from sklearn.metrics import pairwise_distances_chunked
from sklearn.utils import check_X_y,check_random_state
from sklearn.preprocessing import LabelEncoder
import functools
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall
from pyclustering.utils import (draw_clusters,
average_inter_cluster_distance,
average_intra_cluster_distance,
average_neighbor_distance)
import sklearn
from sklearn.metrics import (davies_bouldin_score,
silhouette_score, 
pairwise_distances,
calinski_harabasz_score
)

# They changed the name of calinski_harabaz_score in later version of sklearn:
# https://github.com/scikit-learn/scikit-learn/blob/c4733f4895c1becdf587b38970f6f7066656e3f9/doc/whats_new/v0.20.rst#id2012
sklearn_version = version.parse(sklearn.__version__)
nm_chg_ver = version.parse("0.23")
if sklearn_version >= nm_chg_ver:
    from sklearn.metrics import calinski_harabasz_score as _cal_score
else:
    from sklearn.metrics import calinski_harabaz_score as _cal_score


def _get_clust_pairs(clusters):
    return [(i, j) for i in clusters for j in clusters if i > j]


def _dunn(data=None, dist=None, labels=None):
    clusters = set(labels)
    inter_dists = [
        dist[np.ix_(labels == i, labels == j)].min()
        for i, j in _get_clust_pairs(clusters)
    ]
    intra_dists = [
        dist[np.ix_(labels == i, labels == i)].max()
        for i in clusters
    ]
    return min(inter_dists) / max(intra_dists)

def dunn(dist, labels):
    return _dunn(data=None, dist=dist, labels=labels)


def cop(data, dist, labels):
    clusters = set(labels)
    cpairs = _get_clust_pairs(clusters)
    prox_lst = [
        dist[np.ix_(labels == i[0], labels == i[1])].max()
        for i in cpairs
    ]

    out_l = []
    for c in clusters:
        c_data = data[labels == c]
        c_center = c_data.mean(axis=0, keepdims=True)
        c_intra = pairwise_distances(c_data, c_center).mean()

        c_prox = [prox for pair, prox in zip(cpairs, prox_lst) if c in pair]
        c_inter = min(c_prox)

        to_add = len(c_data) * c_intra / c_inter
        out_l.append(to_add)

    return sum(out_l) / len(labels)


def _silhouette_score2(data=None, dist=None, labels=None):
    return silhouette_score(dist, labels, metric='precomputed')


def _davies_bouldin_score2(data=None, dist=None, labels=None):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'divide by zero')
        return davies_bouldin_score(data, labels)


def _calinski_harabaz_score2(data=None, dist=None, labels=None):
    return _cal_score(data, labels)

def check_number_of_labels(n_labels, n_samples):
    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are 2 "
                         "to n_samples - 1 (inclusive)" % n_labels)

def cluster_dist_reduce(D_chunk, start, labels, label_freqs):
    # accumulate distances from each sample to each cluster
    clust_dists = np.zeros((len(D_chunk), len(label_freqs)),
                           dtype=D_chunk.dtype)
    for i in range(len(D_chunk)):
        clust_dists[i] += np.bincount(labels, weights=D_chunk[i],
                                      minlength=len(label_freqs))

    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (np.arange(len(D_chunk)), labels[start:start + len(D_chunk)])
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]
    # of the remaining distances we normalise and extract the minimum
    clust_dists[intra_index] = np.inf
    clust_dists /= label_freqs
    inter_clust_dists = clust_dists.min(axis=1)
    return intra_clust_dists, inter_clust_dists

def inter_cluster_dist(data=None, dist=None, labels=None):
    _, inter_dist = cluster_distances(dist, labels, metric='precomputed')
    return inter_dist

def intra_cluster_dist(data=None, dist=None, labels=None):
    intra_dist, _ = cluster_distances(dist, labels, metric='precomputed')
    return intra_dist

def cluster_distances(X, labels, *, metric='precomputed', random_state=None, **kwds):
    return intra_inter_distances(X, labels, metric=metric, **kwds)

def intra_inter_distances(X, labels, metric='precomputed'):
    # Check for non-zero diagonal entries in precomputed distance matrix
    atol = np.finfo(X.dtype).eps * 100
    if np.any(np.abs(np.diagonal(X)) > atol):
            raise ValueError(
                'The precomputed distance matrix contains non-zero '
                'elements on the diagonal. Use np.fill_diagonal(X, 0).'
            )

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    check_number_of_labels(len(le.classes_), n_samples)
    reduce_func = functools.partial(cluster_dist_reduce,
                                    labels=labels, label_freqs=label_freqs)
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)
    return np.mean(intra_clust_dists),np.mean(inter_clust_dists)


def clarans_labels(clarans_object):
        labels_clarans = clarans_object.get_clusters()
        labels=pd.DataFrame(labels_clarans).T.melt(var_name='clusters')\
            .dropna()
        labels['value']=labels.value.astype(int)
        labels=labels.sort_values(['value'])\
                .set_index('value')\
                    .values\
                        .flatten()
        return labels

def calculate_clarans_cvi(data,initial_cluster,dist=None):
        cvi_df = pd.DataFrame(columns=['avg_inter_dist','silhouette','calinski',
        'avg_intra_dist','davies','dunn'])
        df_list = data.values.tolist()
        dist=pairwise_distances(data)
        np.fill_diagonal(dist, 0)
        for k in range(initial_cluster,10):
            print(k)
            clarans_model = clarans(df_list,k,3,5)
            (_, result) =timedcall(clarans_model.process)
            labels =  clarans_labels(result)
            # avg_inter_dist = inter_cluster_dist(dist=dist,labels=labels)
            sihlouette = silhouette_score(dist=dist, labels=labels)
            davies = davies_bouldin_score(data, labels)
            calinski = calinski_harabasz_score(data, labels)
            # avg_intra_dist = intra_cluster_dist(dist=dist,labels=labels)
            dunn_ = dunn(dist,labels)
            cvi_df.loc[k] = [avg_inter_dist,sihlouette,
            davies,calinski,avg_intra_dist,dunn_]
            print(cvi_df)
            del clarans_model
        return cvi_df