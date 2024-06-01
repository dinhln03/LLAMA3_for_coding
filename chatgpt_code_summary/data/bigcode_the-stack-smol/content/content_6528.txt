# coding:utf-8

import sys
import codecs
from pathlib import Path
from collections import defaultdict

MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(MAIN_PATH))
from log import log_info as _info
from log import log_error as _error
from log import print_process as _process

class Vertex(object):
  def __init__(self, value):
    self.value = value
  
  def __eq__(self, other):
    return self.value == other.value
  
  def __str__(self):
    return str(self.value)
  
  def __lt__(self, other):
    return self.value < other.value
  
  def __hash__(self):
    return hash(self.value)

class Graph(object):
  def __init__(self):
    self.graph = defaultdict(list)
  
  def addEdge(self, u, v):
    self.graph[u].append(v)
    # for saving the nodes which have no outgoing arc
    if v not in self.graph.keys():
      self.graph[v] = []
    
  def DFSSearchInner(self, u, explored_list):
    explored_list[u] = True
    self.cache.append(u)

    for v in self.graph[u]:
      if not explored_list[v]:
        self.DFSSearchInner(v, explored_list)

  def DFSSearch(self, u):
    explored_list = {}
    for v in self.graph.keys():
      explored_list[v] = False
    self.cache = []
    self.DFSSearchInner(u, explored_list)
    
    return self.cache
  
  def SCCSearch(self, v_sorted):
    self.t = 0
    self.finish_time = {}
    
    explored_list = {}
    for v in self.graph.keys():
      explored_list[v] = False

    leaders = {}
    for v in v_sorted:
      if not explored_list[v]:
        leaders[v] = []
        self.SCCSearch_DFS(v, explored_list, leaders[v])

    return self.finish_time, leaders
  
  def SCCSearch_DFS(self, v, explored_list, leaders):
    explored_list[v] = True
    
    for u in self.graph[v]:
      if not explored_list[u]:
        leaders.append(u)
        self.SCCSearch_DFS(u, explored_list, leaders)
    
    self.t += 1
    self.finish_time[v] = self.t

def readFile(path):
  _info('Start building graph...')

  graph = Graph()
  with codecs.open(path, 'r', 'utf-8') as file:
    data = file.read().split('\n')
  for line in data:
    line_split = line.split(' ')
    u, other = line_split[0], line_split[1:]
    u_obj = Vertex(u)
    for v in other:
      v_obj = Vertex(v)
      graph.addEdge(u_obj, v_obj)

  _info('Finish building graph...')
  return graph

def reverseGraph(graph):
  v_unsorted = list(graph.graph.keys())
  v_sorted = sorted(v_unsorted, reverse=True)

  # reverse the graph
  graph_rev = Graph()
  for v in v_unsorted:
    for u in graph.graph[v]:
      graph_rev.addEdge(u, v)

  return graph_rev, v_sorted

if __name__ == '__main__':
  graph = readFile('test_scc.txt')

  # sanity check
  _info('Check the graph:')
  cache = graph.DFSSearch(Vertex('b'))
  for v in cache:
    print(v, end=' ')
  _info('Finish checking!', head='\n INFO')

  # reverse the graph
  _info('Reverse the graph...')
  graph_rev, v_sorted = reverseGraph(graph)
  
  # sanity check
  _info('Check the graph:')
  cache = graph_rev.DFSSearch(Vertex('a'))
  for v in cache:
    print(v, end=' ')
  _info('Finish checking!', head='\n INFO')
  
  # find SCCs
  finish_time, _ = graph_rev.SCCSearch(v_sorted)
  v_2nd_pass = reversed([v for v, _ in finish_time.items()])
  
  _info('Start finding SCCs...')
  _, leaders = graph.SCCSearch(v_2nd_pass)

  for k in leaders.keys():
    print(k)
  _info('Result:')
  for key, value in leaders.items():
    print(key)
    for v in value:
      print(v)
    print()