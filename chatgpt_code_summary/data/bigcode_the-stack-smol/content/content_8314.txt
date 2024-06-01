"""
Example file on how to display a networkx graph on a browser
"""
import json
import networkx as nx
from networkx.readwrite import json_graph
import http_server
import random


# https://www.alanzucconi.com/2015/11/03/recreational-maths-python/

# Converts a number in the list of its digits
def int_to_list(n):
    # The number is firstly converted into a string using str(n)
    # map -> converts each character of the string into an integer
    return map(int, str(n))

# https://www.alanzucconi.com/2015/11/01/interactive-graphs-in-the-browser/
def toy_graph():
    G = nx.DiGraph()

    for i in range(1, 1000):
        tree = list(set(list(int_to_list(random.randint(1, i)))))

        # Add the entire sequence to the tree
        for j in range(0, len(tree) - 1):
            G.add_edge(tree[j], tree[j + 1])

    for n in G:
        G.node[n]['name'] = n

    d = json_graph.node_link_data(G)
    json.dump(d, open('graph/graph.json', 'w'))

    # The http_server is just a short piece of code that used to be in the
    # examples directory of the networkx library.
    http_server.load_url('graph/graph.html')


if __name__ == '__main__':
    toy_graph()
