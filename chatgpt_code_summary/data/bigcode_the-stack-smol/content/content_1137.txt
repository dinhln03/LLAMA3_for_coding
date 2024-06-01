import functools
import os
import random

import matplotlib.pyplot as plt
import networkx as nx


def make_graph(path):
    G = nx.DiGraph()

    with open(path, 'r') as f:
        lines = f.readlines()
        # random.seed(0)
        sample_nums = int(len(lines) * 0.00006)
        lines = random.sample(lines, sample_nums)
        lines = [line.strip() for line in lines]
        for line in lines:
            edge_node = line.split('	')
            source = int(edge_node[0])
            target = int(edge_node[1])
            G.add_edge(source, target)
    return G


def degree_centrality(G):
    # 节点的度中心性
    if len(G) <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for n, d in G.degree()}
    return centrality


def closeness_centrality(G, u=None, distance=None, wf_improved=True):
    # 节点的接近中心性
    if G.is_directed():
        G = G.reverse()

    if distance is not None:
        path_length = functools.partial(
            nx.single_source_dijkstra_path_length, weight=distance
        )
    else:
        path_length = nx.single_source_shortest_path_length

    if u is None:
        nodes = G.nodes
    else:
        nodes = [u]
    closeness_centrality = {}
    for n in nodes:
        sp = path_length(G, n)
        totsp = sum(sp.values())
        len_G = len(G)
        _closeness_centrality = 0.0
        if totsp > 0.0 and len_G > 1:
            _closeness_centrality = (len(sp) - 1.0) / totsp
            if wf_improved:
                s = (len(sp) - 1.0) / (len_G - 1)
                _closeness_centrality *= s
        closeness_centrality[n] = _closeness_centrality
    if u is not None:
        return closeness_centrality[u]
    else:
        return closeness_centrality


def core_number(G):
    # 节点的核数
    degrees = dict(G.degree())
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    core = degrees
    nbrs = {v: list(nx.all_neighbors(G, v)) for v in G}
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return core


def pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight="weight",
             dangling=None):
    # 节点的pagerank值
    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        s = float(sum(nstart.values()))
        x = {k: v / s for k, v in nstart.items()}

    if personalization is None:
        p = dict.fromkeys(W, 1.0 / N)
    else:
        s = float(sum(personalization.values()))
        p = {k: v / s for k, v in personalization.items()}

    if dangling is None:
        dangling_weights = p
    else:
        s = float(sum(dangling.values()))
        dangling_weights = {k: v / s for k, v in dangling.items()}
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)


def hits(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True):
    # 节点的hub值和authority值
    if len(G) == 0:
        return {}, {}
    if nstart is None:
        h = dict.fromkeys(G, 1.0 / G.number_of_nodes())
    else:
        h = nstart
        s = 1.0 / sum(h.values())
        for k in h:
            h[k] *= s
    for _ in range(max_iter):
        hlast = h
        h = dict.fromkeys(hlast.keys(), 0)
        a = dict.fromkeys(hlast.keys(), 0)
        for n in h:
            for nbr in G[n]:
                a[nbr] += hlast[n] * G[n][nbr].get("weight", 1)
        for n in h:
            for nbr in G[n]:
                h[n] += a[nbr] * G[n][nbr].get("weight", 1)
        s = 1.0 / max(h.values())
        for n in h:
            h[n] *= s
        s = 1.0 / max(a.values())
        for n in a:
            a[n] *= s
        err = sum([abs(h[n] - hlast[n]) for n in h])
        if err < tol:
            break
    else:
        raise nx.PowerIterationFailedConvergence(max_iter)
    if normalized:
        s = 1.0 / sum(a.values())
        for n in a:
            a[n] *= s
        s = 1.0 / sum(h.values())
        for n in h:
            h[n] *= s
    return h, a


def metrics_fuse(G):
    degree = degree_centrality(G)
    closeness = closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)  # 节点的介数中心性
    core = core_number(G)
    pageranks = pagerank(G)
    hubs, authorities = hits(G)

    fused = dict()
    for node in G.nodes:
        deg = degree[node]
        cl = closeness[node]
        bet = betweenness[node]
        co = core[node]
        pr = pageranks[node]
        auth = authorities[node]
        M = 0.05 * deg + 0.15 * cl + 0.1 * bet + 0.3 * co + 0.25 * pr + 0.15 * auth
        fused[node] = M

    pageranks = sorted(pageranks.items(), key=lambda x: x[1], reverse=True)
    print("使用PageRank算法，影响力前10的节点为：")
    for i in range(10):
        print("节点 {}".format(pageranks[i][0]))
    pos = nx.random_layout(G)
    top_nodes = [k for k, v in pageranks[:10]]
    other_nodes = [k for k, v in pageranks[10:]]
    nx.draw_networkx_nodes(G, pos, top_nodes, node_size=200, node_color='Red', alpha=0.6)
    nx.draw_networkx_nodes(G, pos, other_nodes, node_size=200, node_color='Green', alpha=0.6)
    nx.draw_networkx_edges(G, pos)
    labels = dict()
    for k, v in pageranks[:10]:
        labels[k] = k
    nx.draw_networkx_labels(G, pos, labels=labels)
    plt.savefig("./pagerank_result.png")
    plt.show()
    print("---------------------------------------------")

    authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)
    print("使用HITS算法，影响力前10的节点为：")
    for i in range(10):
        print("节点 {}".format(authorities[i][0]))
    pos = nx.random_layout(G)
    top_nodes = [k for k, v in authorities[:10]]
    other_nodes = [k for k, v in authorities[10:]]
    nx.draw_networkx_nodes(G, pos, top_nodes, node_size=200, node_color='Red', alpha=0.6)
    nx.draw_networkx_nodes(G, pos, other_nodes, node_size=200, node_color='Green', alpha=0.6)
    nx.draw_networkx_edges(G, pos)
    labels = dict()
    for k, v in authorities[:10]:
        labels[k] = k
    nx.draw_networkx_labels(G, pos, labels=labels)
    plt.savefig("./hits_result.png")
    plt.show()
    print("---------------------------------------------")

    fused = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    print("使用混合算法，影响力前10的节点为：")
    for i in range(10):
        print("节点 {}".format(fused[i][0]))
    pos = nx.random_layout(G)
    top_nodes = [k for k, v in fused[:10]]
    other_nodes = [k for k, v in fused[10:]]
    nx.draw_networkx_nodes(G, pos, top_nodes, node_size=200, node_color='Red', alpha=0.6)
    nx.draw_networkx_nodes(G, pos, other_nodes, node_size=200, node_color='Green', alpha=0.6)
    nx.draw_networkx_edges(G, pos)
    labels = dict()
    for k, v in fused[:10]:
        labels[k] = k
    nx.draw_networkx_labels(G, pos, labels=labels)
    plt.savefig("./fused_result.png")
    plt.show()
    print("---------------------------------------------")

    return fused


if __name__ == '__main__':
    path = './课程设计数据集.txt'
    if not os.path.exists(path):
        print('未找到数据集')
        exit(1)
    G = make_graph(path)
    metrics_fuse(G)
