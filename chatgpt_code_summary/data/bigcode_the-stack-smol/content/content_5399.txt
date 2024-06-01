"""Training run script"""

import argparse
import json
from pathlib import Path
from bisect import bisect_left

import torch
import torch_geometric as tg
import matplotlib.pyplot as plt
import local2global as l2g

from local2global_embedding.embedding import speye, train, embedding, VGAE_model, VGAE_loss, reconstruction_auc
from local2global_embedding.network import largest_connected_component, TGraph
from local2global_embedding.patches import create_patch_data
from local2global_embedding.clustering import distributed_clustering, fennel_clustering, louvain_clustering, metis_clustering


class ResultsDict:
    """
    Class for keeping track of results
    """
    @classmethod
    def load(cls, filename, replace=False):
        """
        restore results from file

        Args:
            filename: input json file
            replace: set the replace attribute

        Returns:
            populated ResultsDict

        """
        self = cls(replace=replace)
        with open(filename) as f:
            self._data.update(json.load(f))
        return self

    def save(self, filename):
        """
        dump contents to json file

        Args:
            filename: output file path

        """
        with open(filename, 'w') as f:
            json.dump(self._data, f)

    def __init__(self, replace=False):
        """
        initialise empty ResultsDict
        Args:
            replace: set the replace attribute (default: ``False``)
        """
        self._data = {'dims': [], 'auc': [], 'args': []}
        self.replace = replace  #: if ``True``, updates replace existing data, if ``False``, updates append data

    def __getitem__(self, item):
        return self._data[item]

    def _update_index(self, index, aucs: list, args=None):
        """
        update data for a given index

        Args:
            index: integer index into data lists
            aucs: new auc values (should be a list)
            args: new args data (optional)

        """
        if self.replace:
            self['auc'][index] = aucs
            self['args'][index] = args
        else:
            self['auc'][index].extend(aucs)
            self['args'][index].extend([args] * len(aucs))

    def _insert_index(self, index: int, dim: int, aucs: list, args=None):
        """
        insert new data at index

        Args:
            index: integer index into data lists
            dim: data dimension for index
            aucs: new auc values
            args: new args data (optional)
        """
        self['auc'].insert(index, aucs)
        self['dims'].insert(index, dim)
        self['args'].insert(index, [args] * len(aucs))

    def update_dim(self, dim, aucs, args=None):
        """
        update data for given dimension

        Args:
            dim: dimension to update
            aucs: new auc values
            args: new args data (optional)

        if ``self.contains_dim(dim) == True``, behaviour depends on the value of
        ``self.replace``

        """
        index = bisect_left(self['dims'], dim)
        if index < len(self['dims']) and self['dims'][index] == dim:
            self._update_index(index, aucs, args)
        else:
            self._insert_index(index, dim, aucs, args)

    def max_auc(self, dim=None):
        """
        return maximum auc values

        Args:
            dim: if ``dim=None``, return list of values for all dimension, else only return maximum value for ``dim``.

        """
        if dim is None:
            return [max(aucs) for aucs in self['auc']]
        else:
            index = bisect_left(self['dims'], dim)
            if index < len(self['dims']) and self['dims'][index] == dim:
                return max(self['auc'][index])
            else:
                return 0.

    def contains_dim(self, dim):
        """
        equivalent to ``dim in self['dims']``

        """
        index = bisect_left(self['dims'], dim)
        return index < len(self['dims']) and self['dims'][index] == dim

    def reduce_to_dims(self, dims):
        """
        remove all data for dimensions not in ``dims``
        Args:
            dims: list of dimensions to keep

        """
        index = [i for i, d in enumerate(dims) if self.contains_dim(d)]
        for key1 in self._data:
            if isinstance(self._data[key1], list):
                self._data[key1] = [self[key1][i] for i in index]
        return self

    def runs(self, dim=None):
        """
        return the number of runs

        Args:
            dim: if ``dim is None``, return list of number of runs for all dimension, else return number of
                 runs for dimension ``dim``.

        """
        if dim is None:
            return [len(x) for x in self['auc']]
        else:
            index = bisect_left(self['dims'], dim)
            if index < len(self['dims']) and self['dims'][index] == dim:
                return len(self['auc'][index])
            else:
                return 0


_dataloaders = {}  #: dataloaders


def dataloader(name):
    """
    decorator for registering dataloader functions

    Args:
        name: data set name

    """
    def loader(func):
        _dataloaders[name] = func
        return func
    return loader


@dataloader('Cora')
def _load_cora():
    return tg.datasets.Planetoid(name='Cora', root='/tmp/cora')[0]


@dataloader('PubMed')
def _load_pubmed():
    return tg.datasets.Planetoid(name='PubMed', root='/tmp/pubmed')[0]


@dataloader('AMZ_computers')
def _load_amazon_computers():
    return tg.datasets.Amazon(root='/tmp/amazon', name='Computers')[0]


@dataloader('AMZ_photo')
def _load_amazon_photos():
    return tg.datasets.Amazon(root='/tmp/amazon', name='photo')[0]


def load_data(name):
    """
    load data set

    Args:
        name: name of data set (one of {names})

    Returns:
        largest connected component of data set

    """
    data = _dataloaders[name]()
    data = largest_connected_component(data=data)
    data.num_nodes = data.x.shape[0]
    return data


load_data.__doc__ = load_data.__doc__.format(names=list(_dataloaders.keys()))


def prepare_patches(output_folder, **kwargs):
    """
    initialise patch data if ``output_folder`` does not exist, else load existing patch data

    Args:
        output_folder: folder for storing patch data
        **kwargs: arguments passed to :py:func:`~local2global_embedding.patches.create_patch_data`

    Returns:
        patch_data, patch_graph
    """
    output_folder = Path(output_folder)
    if output_folder.is_dir():
        patch_graph = torch.load(output_folder / 'patch_graph.pt')
        patch_data = [torch.load(output_folder / f"patch{i}.pt") for i in range(patch_graph.num_nodes)]
    else:
        patch_data, patch_graph = create_patch_data(**kwargs)
        output_folder.mkdir(parents=True)
        torch.save(patch_graph, output_folder / 'patch_graph.pt')
        for i, data in enumerate(patch_data):
            torch.save(data, output_folder / f'patch{i}.pt')
    return patch_data, patch_graph


def csvlist(input_type=str):
    """
    Create an argparse type that parses comma separated lists of type ``input_type``

    Args:
        input_type: type of list elements

    Returns:
        list parser

    """
    def make_list(input_str):
        return [input_type(s) for s in input_str.split(',')]
    make_list.__doc__ = f"""
    argparse type that parses comma separated list of type {input_type}
    
    Args:
        input_str: string to be parsed
        
    Returns:
        list of elements of type {input_type}
    """
    return make_list


_parser = argparse.ArgumentParser(description="Run training example.")
_parser.add_argument('--data', default='Cora', choices=_dataloaders.keys(), help='Dataset to load')
_parser.add_argument('--no_features', action='store_true', help='Discard features and use node identity.')
_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
_parser.add_argument('--runs', type=int, default=10, help='Number of training runs (keep best result)')
_parser.add_argument('--dims', type=csvlist(int), default=[2], help='Embedding dimensions (comma-separated)')
_parser.add_argument('--hidden_multiplier', type=int, default=2, help='Hidden dim is `hidden_multiplier` * `dim`')
_parser.add_argument('--target_patch_degree', type=float, default=4.0, help='Target patch degree for sparsification.')
_parser.add_argument('--min_overlap', type=int, default=None, help='Minimum target patch overlap (defaults to `max(dims) + 1`)')
_parser.add_argument('--target_overlap', type=int, default=None, help='Target patch overlap (defaults to twice `min_overlap`)')
_parser.add_argument('--gamma', type=float, default=0.0, help="Value of 'gamma' for RMST sparsification.")
_parser.add_argument('--sparsify', default='resistance', help="Sparsification method to use.",
                     choices={'resistance', 'rmst', 'none'})
_parser.add_argument('--cluster', default='metis', choices={'louvain', 'distributed', 'fennel', 'metis'}, help="Clustering method to use")
_parser.add_argument('--num_clusters', default=10, type=int, help="Target number of clusters for fennel, or metis.")
_parser.add_argument('--beta', default=0.1, type=float, help="Beta value for distributed")
_parser.add_argument('--num_iters', default=None, type=int, help="Maximum iterations for distributed or fennel (default depends on method choice)")
_parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
_parser.add_argument('--dist', action='store_true', help='use distance decoder instead of inner product decoder')
_parser.add_argument('--output',
                     default='.',
                     help='output folder')
_parser.add_argument('--device', default=None, help="Device used for training e.g., 'cpu', 'cuda'")
_parser.add_argument('--plot', action='store_true', help='Plot embedding performance')
_parser.add_argument('--verbose', action='store_true', help='Show progress info')


def run(**kwargs):
    """
    Run training example.

    By default this function writes results to the current working directory. To override this use the ``output``
    keyword argument.

    This function reproduces figure 1(a) of [#l2g]_ if called as ``run(dims=[2**i for i in range(1, 8)], plot=True)``.


    Keyword Args:
        data: Name of data set to load (one of {``'Cora'``, ``'PubMed'``, ``'AMZ_computers'``, ``'AMZ_photo'``}) (default: ``'Cora'``)
        no_features: If ``True``, discard features and use node identity. (default: ``False``)
        num_epochs: Number of training epochs (default: ``200``)
        runs: Number of training runs (keep best result) (default: ``1``)
        dims: list of embedding dimensions (default: ``[2]``)
        hidden_multiplier: Hidden dimension is ``hidden_multiplier * dim``
        target_patch_degree: Target patch degree for resistance sparsification. (default: ``4``)
        min_overlap: Minimum target patch overlap (default: ``max(dims) + 1``)
        target_overlap: Target patch overlap (default: ``2 * max(dims)``)
        gamma: Value of 'gamma' for RMST sparsification (default: ``0``)
        sparsify: Sparsification method to use (one of {``'resistance'``, ``'none'``, ``'rmst'``})
                  (default: ``'resistance'``)
        cluster: Clustering method to use (one of {``'louvain'``, ``'fennel'`` , ``'distributed'``, ``'metis'``})
                 (default: ``'metis'``)
        num_clusters: Target number of clusters for distributed, fennel, or metis.
        num_iters: Maximum iterations for distributed or fennel
        lr: Learning rate
        dist: If ``True``, use distance decoder instead of inner product decoder (default: ``False``)
        output: output folder (default: ``'.'``)
        device: Device used for training e.g., 'cpu', 'cuda' (defaults to ``'cuda'`` if available else ``'cpu'``)
        plot: If ``True``, plot embedding performance (default: ``False``)
        verbose: If ``True``, show progress info (default: ``False``)

    This function only accepts keyword arguments and is also exposed as a command-line interface.

    .. rubric:: References

    .. [#l2g] L. G. S. Jeub et al.
              “Local2Global: Scaling global representation learning on graphs via local training”.
              DLG-KDD’21. 2021. `arXiv:2107.12224 [cs.LG] <https://arxiv.org/abs/2107.12224>`_.

    """

    # support calling this as a python function with keyword arguments
    args = _parser.parse_args([])
    for key, value in kwargs.items():
        if key in args:
            setattr(args, key, value)
        else:
            raise TypeError(f'Unknown argument {key}')

    output_folder = Path(args.output)
    data = load_data(args.data)
    neg_edges = tg.utils.negative_sampling(data.edge_index, data.num_nodes)
    graph = TGraph(data.edge_index, data.edge_attr)
    basename = args.data
    dims = args.dims
    num_epochs = args.num_epochs
    runs = args.runs
    min_overlap = args.min_overlap if args.min_overlap is not None else max(dims) + 1
    target_overlap = args.target_overlap if args.target_overlap is not None else 2 * max(dims)

    if args.no_features:
        data.x = None  # remove node features (trained with identity)
        basename += '_no_features'

    if args.dist:
        basename += '_dist'

    if args.sparsify == 'resistance':
        sp_string = f"resistance_deg{args.target_patch_degree}"
    elif args.sparsify == 'rmst':
        sp_string = f"rmst_gamma{args.gamma}"
    elif args.sparsify == 'none':
        sp_string = "no_sparsify"
    else:
        raise RuntimeError(f"Unknown sparsification method '{args.sparsify}'.")
    if args.cluster == 'louvain':
        cluster_fun = lambda: louvain_clustering(graph)
        cluster_string = 'louvain'
    elif args.cluster == 'distributed':
        cluster_fun = lambda: distributed_clustering(graph, args.beta, rounds=args.num_iters)
        cluster_string = f'distributed_beta{args.beta}_it{args.num_iters}'
    elif args.cluster == 'fennel':
        cluster_fun = lambda: fennel_clustering(graph, num_clusters=args.num_clusters, randomise_order=True,
                                                num_iters=args.num_iters)
        cluster_string = f"fennel_n{args.num_clusters}_it{args.num_iters}"
    elif args.cluster == 'metis':
        cluster_fun = lambda: metis_clustering(graph, num_clusters=args.num_clusters)
        cluster_string = f"metis_n{args.num_clusters}"
    else:
        raise RuntimeError(f"Unknown cluster method '{args.cluster}'.")

    cluster_file = output_folder / f"{args.data}_{cluster_string}_clusters.pt"
    if cluster_file.is_file():
        clusters = torch.load(cluster_file)
    else:
        clusters = cluster_fun()
        torch.save(clusters, cluster_file)

    patch_folder = output_folder / f'{args.data}_{cluster_string}_{sp_string}_mo{min_overlap}_to{target_overlap}_patches'
    patch_data, patch_graph = prepare_patches(
        output_folder=patch_folder,
        data=data,
        partition_tensor=clusters,
        min_overlap=min_overlap,
        target_overlap=target_overlap,
        sparsify_method=args.sparsify,
        gamma=args.gamma,
        target_patch_degree=args.target_patch_degree,
        verbose=args.verbose)
    if args.verbose:
        print(f'total edges: {data.num_edges}')
        print(f'total patch edges: {sum(c.num_edges for c in patch_data)}')

    if args.no_features:
        data.x = speye(data.num_nodes)  # add identity as node features for training full model

    # compute baseline full model if necessary
    baseline_file = output_folder / f'{basename}_full_info.json'
    training_args = {'lr': args.lr, 'num_epochs': args.num_epochs, 'hidden_multiplier': args.hidden_multiplier}
    if baseline_file.is_file():
        baseline_data = ResultsDict.load(baseline_file)
    else:
        baseline_data = ResultsDict()

    for d in dims:
        r = baseline_data.runs(d)
        if r < runs:
            if args.verbose:
                print(f'training full model for {runs-r} runs and d={d}')
            for r_it in range(r, runs):
                if args.verbose:
                    print(f"full model (d={d}) run {r_it + 1} of {runs}")
                data = data.to(args.device)
                model = train(data,
                              VGAE_model(d, d * args.hidden_multiplier, data.num_features, dist=args.dist).to(args.device),
                              loss_fun=VGAE_loss,
                              num_epochs=num_epochs,
                              lr=args.lr,
                              verbose=args.verbose,
                              )
                coords = embedding(model, data)
                auc = reconstruction_auc(coords, data, dist=args.dist)
                if auc > baseline_data.max_auc(d):
                    if args.verbose:
                        print(f"new best (auc={auc})")
                    torch.save(model.state_dict(), output_folder / f'{basename}_full_d{d}_best_model.pt')
                    torch.save(coords, output_folder / f'{basename}_full_d{d}_best_coords.pt')
                baseline_data.update_dim(d, [auc], training_args)
                baseline_data.save(baseline_file)

    results_file = patch_folder / f'{basename}_l2g_info.json'
    nt_results_file = patch_folder / f'{basename}_nt_info.json'
    if results_file.is_file():
        results = ResultsDict.load(results_file, replace=True)
    else:
        results = ResultsDict(replace=True)
    if nt_results_file.is_file():
        nt_results = ResultsDict.load(nt_results_file, replace=True)
    else:
        nt_results = ResultsDict(replace=True)

    for d in dims:
        patch_list = []
        update_aligned_embedding = False
        for p_ind, patch in enumerate(patch_data):
            patch_result_file = patch_folder / f'{basename}_patch{p_ind}_info.json'
            if patch_result_file.is_file():
                patch_results = ResultsDict.load(patch_result_file)
            else:
                patch_results = ResultsDict()
            coords_file = patch_folder / f'{basename}_patch{p_ind}_d{d}_best_coords.pt'
            if coords_file.is_file():
                best_coords = torch.load(coords_file)

            r = patch_results.runs(d)
            if args.no_features:
                patch.x = speye(patch.num_nodes)
            if r < runs:
                if args.verbose:
                    print(f'training patch{p_ind} for {runs-r} runs and d={d}')
                patch = patch.to(args.device)
                for r_it in range(r, runs):
                    if args.verbose:
                        print(f"patch{p_ind} (d={d}) run {r_it+1} of {runs}")
                    model = train(patch,
                                  VGAE_model(d, d * args.hidden_multiplier, patch.num_features, dist=args.dist).to(args.device),
                                  loss_fun=VGAE_loss,
                                  num_epochs=num_epochs,
                                  lr=args.lr,
                                  )
                    coords = embedding(model, patch)
                    auc = reconstruction_auc(coords, patch, dist=args.dist)
                    if auc > patch_results.max_auc(d):
                        if args.verbose:
                            print(f"new best (auc={auc})")
                        best_coords = coords
                        torch.save(model.state_dict(), patch_folder / f'{basename}_patch{p_ind}_d{d}_best_model.pt')
                        torch.save(best_coords, coords_file)
                        update_aligned_embedding = True
                    patch_results.update_dim(d, [auc], training_args)
                    patch_results.save(patch_result_file)
            patch_list.append(l2g.Patch(patch.nodes.cpu().numpy(), best_coords.cpu().numpy()))


        patched_embedding_file = patch_folder / f'{basename}_d{d}_coords.pt'
        patched_embedding_file_nt = patch_folder / f'{basename}_d{d}_ntcoords.pt'
        if update_aligned_embedding or not patched_embedding_file.is_file():
            prob = l2g.WeightedAlignmentProblem(patch_list, patch_edges=patch_graph.edges())
            ntcoords = prob.mean_embedding()
            coords = prob.get_aligned_embedding()
            torch.save(coords, patched_embedding_file)
            torch.save(ntcoords, patched_embedding_file_nt)

            results.update_dim(d, [reconstruction_auc(torch.as_tensor(coords), data, neg_edges, dist=args.dist)])
            nt_results.update_dim(d, [reconstruction_auc(torch.as_tensor(ntcoords), data, neg_edges, dist=args.dist)])
            results.save(results_file)
            nt_results.save(nt_results_file)

    baseline_data = baseline_data.reduce_to_dims(dims)
    results = results.reduce_to_dims(dims)
    nt_results = nt_results.reduce_to_dims(dims)

    if args.plot:
        plt.figure()
        plt.plot(dims, [max(v) for v in baseline_data['auc']], label='full, inner product', marker='o',
                 color='tab:blue')
        plt.plot(dims, results['auc'], '--', label='l2g, inner product', marker='>', color='tab:blue')
        plt.plot(dims, nt_results['auc'], ':', label='no-trans, inner product', color='tab:blue',
                 linewidth=1)

        plt.xscale('log')
        plt.xticks(dims, dims)
        plt.minorticks_off()
        plt.xlabel('embedding dimension')
        plt.ylabel('AUC')
        plt.legend()
        oversampling_ratio = sum(p.num_edges for p in patch_data) / data.num_edges
        plt.title(f"oversampling ratio: {oversampling_ratio:.2}, #patches: {len(patch_data)}")
        plt.savefig(output_folder / f"{basename}_{cluster_string}_{sp_string}_mo{min_overlap}_to{target_overlap}.pdf")
        plt.show()


if __name__ == '__main__':
    # run main script
    args = _parser.parse_args()
    run(**vars(args))
