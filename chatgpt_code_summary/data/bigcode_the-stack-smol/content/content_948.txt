from math import sqrt

import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout


class Distribution:
    from random import random
    from random import gauss
    from numpy.random import poisson

    _h = [0]
    h = property(lambda s: s._h[0])

    drop_rate = 0
    move_rate = 0
    move_int = 600
    tx_rate = 0
    em_rate = 0
    aw_rate = lambda s, n: 0

    @classmethod
    def aloha(cls, k, n):
        r = cls.random()
        return r

    @classmethod
    def tx_chn(cls, a, g):
        return 0

    @classmethod
    def tx_awt(cls, a, g):
        global awt
        fold = sum(p.timeout for p in a.buffer)
        return fold + cls.aw_rate(len(b.children))

    @classmethod
    def emit(cls, k):
        return cls.poisson(cls.em_rate*k)

    @classmethod
    def tx(cls, a, b, g):
        return cls.tx_awt(a, b, g) + cls.tx_chn(a, b, g)

    @classmethod
    def mv(cls):
        if cls.random() < cls.move_rate:
            return cls.random()*cls.move_int

    @classmethod
    def drop(cls):
        return cls.random() < cls.drop_rate


class Abonent(Distribution):
    drop_rate = 1e-8
    move_rate = 0
    aw_rate = 1.0/1e9
    em_rate = property(lambda s: s.h/100.0)


class MobileAbonent(Abonent):
    move_rate = 0.5


class Operator(Distribution):
    drop_rate = 1e-8
    move_rate = 0
    aw_rate = 1.0/1e10
    em_rate = 0


class Server(Distribution):
    drop_rate = 1e-8
    move_rate = 0
    aw_rate = 1.0/5e9
    em_rate = property(lambda s: s.h/100.0)


class WiFi(Distribution):
    mu, sigma = 2e-6, 1e-6

    drop_rate = 0.005
    tx_rate = 0.1
    aw_rate = lambda s, n: s.aloha(s.mu, n)


class Fiber(Distribution):
    mu, sigma = 2e-8, 1e-8

    drop_rate = 1e-12
    tx_rate = 10
    aw_rate = lambda s, n: s.aloha(s.mu, n)


class Ethernet(Distribution):
    mu = 2e-7

    drop_rate = 1e-10
    tx_rate = property(lambda s: 6 - s.random()*5)
    aw_rate = lambda s, n: s.aloha(s.mu, 2)


class LTE(Distribution):
    mu, sigma = 2e-7, 1e-7

    drop_rate = 1e-10
    tx_rate = property(lambda s: 6 - s.random()*5)
    aw_rate = lambda s, n: s.gauss(s.mu*n, s.sigma*sqrt(n))


    
class Node:
    def __init__(self, id, g):
        self.id = id
        self.g = g

    def __getattr__(self, key):
        return self.g.node[self.id][key]

    @property
    def buffer(self):
        return filter(lambda p: p.curr == self, map(lambda e: e.obj, self.g.events))


class Graph(nx.DiGraph):
    c = root = 12007

    def iterate(self, r, n, d, node, channel):
        for _ in xrange(0, n):
            self.c += 1
            self.add_node(self.c, deep=d, distr=node)
            self.add_edge(r, self.c, distr=channel)
            self.add_edge(self.c, r, distr=Ethernet)
            yield self.c

    def paths(self, a, b):
        return self.all_shortest_paths(a.id, b.id)

    def __init__(self, deep=5, icount=3, operators=10):
        nx.DiGraph.__init__(self)
        q = [self.root + i for i in xrange(0, operators)]
        self.c += operators - 1
        self.deep = deep
        for r in q:
            self.add_node(r, distr=Operator, deep=0)
        if operators > 1:
            for u, v in zip(q[1:], q[:-1]):
                self.add_edge(u, v, distr=Fiber)
        for deep in xrange(1, deep+1):
            q, last = [], q
            for r in last:
                for v in self.iterate(r, icount + 1 if deep == self.deep else icount, deep, Operator, Ethernet):
                    q.append(v)

    @property
    def operators(self):
        return filter(lambda x: self.node[x]["deep"] != self.deep, self.nodes())

    @property
    def leaves(self):
        return filter(lambda x: self.node[x]["deep"] == self.deep, self.nodes())
        
    def show(self):
        print len(self.nodes())
        pos = graphviz_layout(self, prog="sfdp", args="")
        plt.rcParams["axes.facecolor"] = "black"
        nx.draw_networkx_nodes(self, pos, nodelist=self.operators, node_color="gray", node_size=10)
        nx.draw_networkx_nodes(self, pos, nodelist=self.leaves, node_color="red", node_size=10)
        nx.draw_networkx_edges(self, pos, edge_color="white", arrows=False)
        plt.show()


if __name__ == "__main__":
    Graph().show()