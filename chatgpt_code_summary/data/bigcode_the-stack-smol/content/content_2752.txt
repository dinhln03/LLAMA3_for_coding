import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(cell_line, cross_cell_line, label_rate, k_mer):
    """
    Load input data from data/cell_line directory.

    | x_20.index   | the indices (IDs) of labeled train instances as list object (for label_rate = 20%) |
    | ux_20.index  | the indices (IDs) of unlabeled train instances as list object (for label_rate = 20%) |
    | vx_20.index  | the indices (IDs) of validation instances as list object (for label_rate = 20%) |
    | tx_20.index  | the indices (IDs) of test instances as list object (for label_rate = 20%) |
    | features_5mer | the feature vectors of all instances as scipy.sparse.csr.csr_matrix object (for k_mer = 5) |
    | nodes         | a dict in the format {chromosome_name: ID} as collections.defaultdict object |
    | labels        | the one-hot labels of all instances as numpy.ndarray object |
    | graph         | a dict in the format {ID: [IDs_of_neighbor_nodes]} as collections.defaultdict object |

    All objects above must be saved using python pickle module.

    :param cell_line: Name of the cell line to which the datasets belong
    :return: All data input files loaded (as well the training/test data).
    """

    if (cross_cell_line != None) and (cross_cell_line != cell_line):
        read_dir = 'data/{}_{}/'.format(cell_line, cross_cell_line)
    else:
        read_dir = 'data/{}/'.format(cell_line)

    # STEP 1: Load all feature vectors, class labels and graph
    features_file = open('{}/features_{}mer'.format(read_dir, k_mer), "rb")
    features = pkl.load(features_file)
    features_file.close()

    labels_file = open('{}/labels'.format(read_dir), "rb")
    labels = pkl.load(labels_file)
    labels_file.close()

    graph_file = open('{}/graph'.format(read_dir), "rb")
    graph = pkl.load(graph_file)
    graph_file.close()

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # STEP 2: Load IDs of labeled_train/unlabeled_train/validation/test nodes
    lr = txt = '{:.2f}'.format(label_rate).split('.')[1]

    idx_x_file = open('{}/x_{}.index'.format(read_dir, lr), "rb")
    idx_x = pkl.load(idx_x_file)
    idx_x_file.close()

    idx_ux_file = open('{}/ux_{}.index'.format(read_dir, lr), "rb")
    idx_ux = pkl.load(idx_ux_file)
    idx_ux_file.close()

    idx_vx_file = open('{}/vx_{}.index'.format(read_dir, lr), "rb")
    idx_vx = pkl.load(idx_vx_file)
    idx_vx_file.close()

    idx_tx_file = open('{}/tx_{}.index'.format(read_dir, lr), "rb")
    idx_tx = pkl.load(idx_tx_file)
    idx_tx_file.close()

    # STEP 3: Take subsets from loaded features and class labels using loaded IDs
    x = features[idx_x]
    y = labels[idx_x]

    ux = features[idx_ux]
    uy = labels[idx_ux]

    vx = features[idx_vx]
    vy = labels[idx_vx]

    tx = features[idx_tx]
    ty = labels[idx_tx]

    print("x={} ux={} vx={} tx={}".format(x.shape[0], ux.shape[0], vx.shape[0], tx.shape[0]))

    # STEP 4: Mask labels
    train_mask = sample_mask(idx_x, labels.shape[0])
    val_mask = sample_mask(idx_vx, labels.shape[0])
    test_mask = sample_mask(idx_tx, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
