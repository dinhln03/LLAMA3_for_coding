# -*- coding: utf-8 -*-
#    Copyright (C) 2018 by
#    Marta Grobelna <marta.grobelna@rwth-aachen.de>
#    Petre Petrov <petrepp4@gmail.com>
#    Rudi Floren <rudi.floren@gmail.com>
#    Tobias Winkler <tobias.winkler1@rwth-aachen.de>
#    All rights reserved.
#    BSD license.
#
# Authors:  Marta Grobelna <marta.grobelna@rwth-aachen.de>
#           Petre Petrov <petrepp4@gmail.com>
#           Rudi Floren <rudi.floren@gmail.com>
#           Tobias Winkler <tobias.winkler1@rwth-aachen.de>

import random as rnd
from collections import deque

import networkx as nx

from planar_graph_sampler.combinatorial_classes.half_edge_graph import HalfEdgeGraph


class IrreducibleDissection(HalfEdgeGraph):
    """
    Represents the class 'I' of irreducible dissections from the paper.
    It is however also used for rooted and derived dissections (sizes are incorrect then).

    Parameters
    ----------
    half_edge: ClosureHalfEdge
        A half-edge on the hexagonal boundary of a closed binary tree.
    """

    def __init__(self, half_edge):
        assert half_edge.is_hexagonal
        if half_edge.color is not 'black':
            half_edge = half_edge.opposite.next
        assert half_edge.color is 'black'
        super(IrreducibleDissection, self).__init__(half_edge)

    @property
    def is_consistent(self):
        super_ok = super(IrreducibleDissection, self).is_consistent
        root_is_black = self.half_edge.color is 'black'
        root_is_hex = self.half_edge.is_hexagonal
        twelve_hex_he = len([he for he in self.half_edge.get_all_half_edges() if he.is_hexagonal]) == 12
        return all([super_ok, root_is_black, root_is_hex, twelve_hex_he, self.is_admissible])

    @property
    def hexagonal_edges(self):
        """Gets the three half-edges on the hexagonal boundary incident to a black node and point in ccw direction."""
        first = self.half_edge
        res = [first]
        second = first.opposite.next.opposite.next
        res.append(second)
        third = second.opposite.next.opposite.next
        res.append(third)
        for he in res:
            assert he.is_hexagonal and he.color is 'black'
        return res

    def root_at_random_hexagonal_edge(self):
        """Selects a random hexagonal half-edge and makes it the root."""
        self._half_edge = rnd.choice(self.hexagonal_edges)

    @property
    def is_admissible_slow(self):
        """Checks if there is a path of length 3 with an inner edge from the root to the opposite outer vertex."""

        start_node = self.half_edge
        assert start_node.color is 'black'
        end_node = self.half_edge.opposite.next.opposite.next.opposite
        assert end_node.color is 'white'
        start_node = start_node.node_nr
        end_node = end_node.node_nr

        g = self.to_networkx_graph()

        # There are always 2 path of length 4 (meaning 4 nodes) from start to end (on the hexagon boundary).
        # If there is one more, then this is a forbidden path!
        paths = nx.shortest_simple_paths(g, start_node, end_node)
        path_1 = next(paths)
        assert len(path_1) == 4
        path_2 = next(paths)
        assert len(path_2) == 4
        path_3 = next(paths)
        return len(path_3) > 4


    @property
    def is_admissible(self):
        """Checks if there is a path of length 3 with an inner edge from the root to the opposite outer vertex."""

        start_node = self.half_edge
        assert start_node.color is 'black'
        end_node = self.half_edge.opposite.next.opposite.next.opposite
        assert end_node.color is 'white'

        # Creates the queue for the BFS.
        queue = deque(list())
        # Put the init half edge into the queue.
        queue.append((self.half_edge, 0, False, set()))

        while len(queue) != 0:
            # Pop the _first element from the FIFO queue.
            top_element = queue.popleft()

            # Extract the components from the top element.
            top_half_edge = top_element[0]
            distance = top_element[1]
            has_been_inner_edge_included = top_element[2]
            visited_nodes = top_element[3]

            # Updated the visited_nodes_set.
            visited_nodes.add(top_half_edge.node_nr)

            # Start BFS for the half edges connected with the specific node.
            incident_half_edges = top_half_edge.incident_half_edges()
            for walker_half_edge in incident_half_edges:

                opposite = walker_half_edge.opposite
                # Skip the vertex if it was already visited.
                if opposite in visited_nodes: continue

                # Prepare the new components of the element.
                updated_distance = distance + 1
                new_visited_nodes = set()
                new_visited_nodes.update(visited_nodes)
                inner_edge_included = has_been_inner_edge_included or (opposite.is_hexagonal is False)

                # If the distance is smaller than 3 then the element is added into the queue.
                if updated_distance < 3:
                    queue.append((opposite, updated_distance, inner_edge_included, new_visited_nodes))
                else:
                    # If the distance is equal to 3 than we check whether the new vertex is the outer one and
                    # does an inner edge have been included in the path. If both conditions are True, then a path
                    # has been found which means that the dissection is not irreducible. -> Return false.
                    if opposite.node_nr == end_node.node_nr and inner_edge_included:
                        return False

        # A path has not been found, therefore the dissection is irreducible and we return True.
        return True

    # CombinatorialClass interface.

    @property
    def u_size(self):
        """The u-size is the number of inner faces."""
        return (self.number_of_half_edges - 6) / 4

    @property
    def l_size(self):
        """The l-size is the number of black inner vertices."""
        node_dict = self.half_edge.node_dict()
        black_vertices = len([node_nr for node_nr in node_dict if node_dict[node_nr][0].color is 'black'])
        # There are always 3 hexagonal outer black vertices.
        return black_vertices - 3

    # Networkx related functionality.

    def to_networkx_graph(self, include_unpaired=None):
        """Converts to networkx graph, encodes hexagonal nodes with colors."""
        from planar_graph_sampler.combinatorial_classes.half_edge_graph import color_scale
        # Get dict of nodes.
        nodes = self.half_edge.node_dict()
        # Include the leaves as well.
        G = super(IrreducibleDissection, self).to_networkx_graph(include_unpaired=False)
        for v in G:
            if any([he.is_hexagonal for he in nodes[v]]):
                G.nodes[v]['color'] = '#e8f442'
            else:
                G.nodes[v]['color'] = '#aaaaaa'
            if nodes[v][0].color is 'black':
                # Make black nodes darker.
                G.nodes[v]['color'] = color_scale(G.nodes[v]['color'], 0.5)
        return G
