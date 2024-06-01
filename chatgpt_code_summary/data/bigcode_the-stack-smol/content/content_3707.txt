from collections import defaultdict

from hsst.utility import search
from hsst.utility.graph import SemanticGraph


class SubgraphEnumeration(object):

    def __init__(self, graph, node_set_size_limit=0):

        self.full_node_set = graph.nodes
        self.full_edge_set = graph.edges
        self.current_node_set = set()
        self.current_edge_set = set()
        self.visited_states = set()
        self.subgraphs = []

        self.node_set_size_limit = node_set_size_limit

        # Create fast lookup structures
        self.edges_by_source = defaultdict(set)
        self.edges_by_destination = defaultdict(set)
        self.edges_by_both = defaultdict(set)
        self.labels = defaultdict(list)

        for edge in self.full_edge_set:
            self.labels[(edge.from_node, edge.to_node)].append(edge)
            self.edges_by_source[edge.from_node].add(edge.to_node)
            self.edges_by_destination[edge.to_node].add(edge.from_node)
            self.edges_by_both[edge.from_node].add(edge.to_node)
            self.edges_by_both[edge.to_node].add(edge.from_node)

    def generate_moves(self):
        # Generate all possible moves

        # Each move consists of a single node and the set of edges that connect that node to the nodes
        # in the currentNodeSet E.g. ( node, { (label1, node, node1), (label2, node2, node) ... } )
        # Moves are temporarily stored as a dictionary so that the full set of edges associated with each move
        # can be constructed

        moves = []
        temporary_moves = {}

        # Check if the limit for the currentNodeSet size has been reached
        if 0 < self.node_set_size_limit <= len(self.current_node_set):
            return moves

        # The initial step is handled separately
        if not self.current_node_set:
            for node in self.full_node_set:
                moves.append((node, set()))

            return moves

        # The set of possible nodes consists of nodes that are not yet in the currentNodeSet
        possible_nodes = self.full_node_set - self.current_node_set

        # For every possible node, we need to check that it shares an edge with a node in the currentNodeSet
        # Otherwise we would violate the 'connected' constraint

        for possible_node in possible_nodes:

            destination_nodes = self.edges_by_source[possible_node] & self.current_node_set
            source_nodes = self.edges_by_destination[possible_node] & self.current_node_set

            if len(destination_nodes) > 0 or len(source_nodes) > 0:
                # There is at least one node in the current node set that we can connect the possible_node to
                # Check if this state has been explored already
                if self.id(node=possible_node) in self.visited_states:
                    continue

                # If not, it is an acceptable move and we just need to construct the edge set that connects
                # the possible_node to the current node set

                edges = set(
                    edge for source_node in source_nodes for edge in self.labels[(source_node, possible_node)]) | \
                        set(edge for destination_node in destination_nodes for edge in
                            self.labels[(possible_node, destination_node)])

                temporary_moves[possible_node] = edges

        for move in temporary_moves:
            moves.append((move, temporary_moves[move]))

        return moves

    def move(self, move):
        # Move is a tuple (node, edge_set)
        node, edge_set = move

        self.current_node_set.add(node)
        self.current_edge_set |= edge_set
        self.visited_states.add(self.id())

        self.subgraphs.append((self.current_node_set.copy(), self.current_edge_set.copy()))

    def undo_move(self, move):
        # Move is a tuple (node,         edge_set)
        node, edge_set = move

        self.current_node_set.remove(node)
        self.current_edge_set -= edge_set

    def solved(self):
        return False

    def id(self, node=None):
        if node:
            return " ".join(str(x) for x in sorted(self.current_node_set | {node}, key=lambda x: x.node_id))
        else:
            return " ".join(str(x) for x in sorted(self.current_node_set, key=lambda x: x.node_id))


def enumerate_dfs_subgraphs(graph, df_limit=100):
    enumeration = SubgraphEnumeration(graph, node_set_size_limit=df_limit)
    search.df(enumeration, df_limit)
    return set(SemanticGraph(nodes, edges, nonterminal_count=0) for nodes, edges in enumeration.subgraphs)
