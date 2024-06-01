#
# This example is again a graph coloring problem. In this case, however,
# a stronger object oriented approach is adopted to show how Coopy is
# indeed compatible with such practices.
#

import coopy
import random

class Node:

    def __init__(self):
        self._color = coopy.symbolic_int('c')
        self._neighbors = set()

    @property
    def color(self):
        return self._color

    @property
    def has_valid_connections(self):
        return coopy.all([self.color != n.color for n in self._neighbors])

    def direct_edge_towards(self, other):
        self._neighbors.add(other)

    def __repr__(self):
        return str(self.color)

def construct_k_colored_graph(k, n, p):
    """
    Constructs a k colored graph of n nodes in which a pair
    of nodes shares an edge with probability 0 <= p <= 1.

    Note: this code is for demonstrative purposes only; the
    solution for such a problem will not necessarily exist,
    in which case the concretization process will throw
    an exception.
    """
    with coopy.scope():
        
        # Instantiate n nodes.
        nodes = [Node() for i in range(n)]

        # Connect nodes with probability p.
        for i in range(n-1):
            for j in range(i+1,n):
                a = nodes[i]
                b = nodes[j]
                if random.uniform(0,1) < p:
                    a.direct_edge_towards(b)
                    b.direct_edge_towards(a)

        # Impose restrictions over the nodes.
        for node in nodes:
            coopy.any([node.color == i for i in range(k)]).require()
            node.has_valid_connections.require()

        # Concretize the graph and return it as a list of nodes.
        coopy.concretize()
        return nodes

graph = construct_k_colored_graph(3, 10, 0.2)
print(graph)