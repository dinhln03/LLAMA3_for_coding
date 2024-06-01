import time
from grapht.graph import DictGraph, StreamGraph

__author__ = 'willmcginnis'

if __name__ == '__main__':
    print('DictGraph')
    g = {0: [2], 1: [2, 3], 2: [1, 3], 3: [1, 2, 4, 6], 4: [3, 5], 5: [4, 6, 7], 6: [3, 4, 5], 7: [5, 8], 8: [7], 9: [8], 10: [9], 11: [8, 9], 12: [11], 13: [12], 14: [13], 15: [1]}
    gp = DictGraph(g)

    print('Original Adjacency Matrix')
    print(gp.get_dense())

    print('Second Connections')
    print(gp.get_n_connection(n=2).toarray())

    print('Third Connections')
    print(gp.get_n_connection(n=3).toarray())

    print('\n\nStream Graph')

    # NOTE: You'll need a graph in a postgres db to actually do this.
    gp2 = StreamGraph(max_dim=28000000)
    gp2.from_psql(username='postgres',
                  password='admin',
                  database='',
                  host='localhost',
                  schema='directed',
                  table='graph')

    print('Number of non-zero elements')
    edges = gp2.get_nnz()
    print(edges)

    print('Calculating 2nd Degree connections for a %s edge graph' % (edges, ))
    start_time = time.time()
    temp = gp2.get_n_connection(n=2)
    elapsed = time.time() - start_time
    print('TIME: %s' % (str(elapsed), ))

    print('\nMost Connected N')
    res = gp2.most_connected_n(n=25)
    print(res)