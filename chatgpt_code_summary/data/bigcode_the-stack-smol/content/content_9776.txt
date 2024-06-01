import networkx
import random

def regularize_graph(graph,d):
  regularized = True
  for node_id in list(graph.nodes()):
    if graph.in_degree(node_id)!=d or graph.out_degree(node_id)!=d:
      regularized = False
      break
  while not regularized:
    lost_in_degree_ids = []
    full_in_degree_ids = []
    for node_id in list(graph.nodes()):
      if graph.in_degree(node_id)<d:
        lost_in_degree_ids.append(node_id)
      elif graph.in_degree(node_id)==d:
        full_in_degree_ids.append(node_id)
      else:
        raise Exception('In degree too large')
    lost_in_degree_ids = random.sample(lost_in_degree_ids, len(lost_in_degree_ids))
    lost_outdegree_ids = []
    full_outdegree_ids = []
    for node_id in list(graph.nodes()):
      if graph.out_degree(node_id)<d:
        lost_outdegree_ids.append(node_id)
      elif graph.out_degree(node_id)==d:
        full_outdegree_ids.append(node_id)
      else:
        raise Exception('Out degree too large')
    lost_outdegree_ids = random.sample(lost_outdegree_ids, len(lost_outdegree_ids))
    if len(lost_in_degree_ids)!=len(lost_outdegree_ids):
      raise Exception('Number of missing in and out degrees do not match')
    for i in range(len(lost_in_degree_ids)):
      full_in_degree_ids = random.sample(full_in_degree_ids, len(full_in_degree_ids))
      full_outdegree_ids = random.sample(full_outdegree_ids, len(full_outdegree_ids))
      lost_in_degree_id = lost_in_degree_ids[i]
      lost_outdegree_id = lost_outdegree_ids[i]
      # Find appropriate (full_outdegree_id, full_in_degree_id) pair
      full_in_degree_id = -1
      full_outdegree_id = -1
      for fod_id in full_outdegree_ids:
        if fod_id!=lost_in_degree_id:
          suc_ids = list(graph.successors(fod_id))
          for suc_id in suc_ids:
            if (suc_id in full_in_degree_ids) and (suc_id!=lost_outdegree_id):
              full_in_degree_id = suc_id
              full_outdegree_id = fod_id
              break
          if full_in_degree_id!=-1 and full_outdegree_id!=-1:
            break
      # Patch
      graph.remove_edge(full_outdegree_id, full_in_degree_id)
      graph.add_edge(full_outdegree_id, lost_in_degree_id)
      graph.add_edge(lost_outdegree_id, full_in_degree_id)
    regularized = True
    for node_id in list(graph.nodes()):
      if graph.in_degree(node_id)!=d or graph.out_degree(node_id)!=d:
        regularized = False
        break
  return graph
