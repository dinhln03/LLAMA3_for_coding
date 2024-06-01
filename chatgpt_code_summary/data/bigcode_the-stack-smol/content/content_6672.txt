import os
import sys
import math
import copy

from binary_tree import BinaryTreeNode, BinaryTree, BinarySearchTree
from graph import GraphNode, Graph


# 4.6 find the next node (in-order) of a given node in a Binary Tree
# -> back to root and using in-order travelsal until meet the current node. get the next
def get_next_node(node):
  root = get_root_node(node)
  is_next = [False]
  next_node = get_next_node_in_order_of_node(node, root, is_next)
  return next_node

def get_next_node_in_order_of_node(node, visit_node, is_next):

  if is_next[0]:
    return visit_node
    
  if visit_node == None:
    return None

  node_next = get_next_node_in_order_of_node(node, visit_node.left, is_next)
  if node_next != None:
    return node_next
  
  if is_next[0]:
    return visit_node

  if visit_node == node:
    is_next[0] = True

  node_next = get_next_node_in_order_of_node(node, visit_node.right, is_next)
  if node_next != None:
    return node_next
  return None


def get_root_node(node):
  root = node
  while node.parent != None:
    node = node.parent
  return node


# Test
# array = [1,2,3,4,5,6]
# tree = BinaryTree()
# for v in array:
#   tree.append(v)
# node = tree.root.left.right
# next_node = get_next_node(node)
# if next_node != None:
#   print(next_node.value)
# else:
#   print("None")


# 4.7 build projects
class Project:
  name = ""
  dependencies = list() # list of dependency projects
  state = 0 # 0: waiting, 1: built
  def __init__(self, name):
    self.name = name
    self.state = 0
    self.dependencies = list()

def build_projects(projects):
  build_queue = list()
  
  while True:
    has_new_build = False
    for project in projects:
      if project.state == 0:
        if build_project(project) == True:
          build_queue.append(project.name)
          project.state = 1
          has_new_build = True
    if has_new_build == False:
      break
  
  is_built_all = True
  for project in projects:
    if project.state == 0:
      is_built_all = False
      break
  
  if is_built_all: 
    return build_queue
  else:
    return False


def build_project(project):
  is_dependencies_built = True
  for dep in project.dependencies:
    if dep.state != 1:
      is_dependencies_built = False
      break  
  if is_dependencies_built:
    project.state = 1
    return True
  else:
    return False


# a = Project("a")
# b = Project("b")
# c = Project("c")
# d = Project("d")
# e = Project("e")
# f = Project("f")
# d.dependencies.append(a)
# b.dependencies.append(f)
# d.dependencies.append(b)
# a.dependencies.append(f)
# c.dependencies.append(d)
# t = build_projects([a,b,c,d,e,f])
# print(t)


# 4.8 find first common ancestor
# -> get a queue ancestor of node 1 and compare for node 2
def get_common_ancestor(node1, node2):
  if node1 == node2:
    return node1
  
  node1_parents = list()
  parent1 = node1
  while parent1 != None:
    node1_parents.append(parent1)
    parent1 = parent1.parent
  
  node2_parents = list()
  parent2 = node2
  while parent2 != None:
    node2_parents.append(parent2)
    parent2 = parent2.parent

  common_ancestor = None
  for p1 in node1_parents:
    for p2 in node2_parents:
      if p1 == p2:
        common_ancestor = p1
        break
    if common_ancestor != None:
      break

  return common_ancestor
  

# Test
# array = [1,2,3,4,5,6]
# tree = BinaryTree()
# for v in array:
#   tree.append(v)
# n1 = tree.root.left.left
# n2 = tree.root.right.left
# common = get_common_ancestor(n1, n2)
# print(common.value)


# 4.9 print all possible array can be create from a binary search tree
def dump_permutation_of_source_array(tree):
  if tree.root != None:
    _dump_permutation_of_source_array([tree.root], [])
  else:
    print("tree is empty")


def _dump_permutation_of_source_array(candidate_nodes, visited_nodes):
  
  if len(candidate_nodes) == 0:
    dump_nodes(visited_nodes)
    return

  n = len(candidate_nodes)
  for i in range(0, n):
    
    _visited_nodes = copy.deepcopy(visited_nodes)
    _candidate_nodes = copy.deepcopy(candidate_nodes)

    _visited_nodes.append(_candidate_nodes[i])
    _candidate_nodes.remove(_candidate_nodes[i])

    node = candidate_nodes[i]
    if node.left != None:
      _candidate_nodes.insert(0, node.left)
    if node.right != None:
      _candidate_nodes.insert(0, node.right)

    _dump_permutation_of_source_array(_candidate_nodes, _visited_nodes)
  

def dump_nodes(nodes):
  values = []
  for node in nodes:
    values.append(node.value)
  print("source:", values)


# Test
# values = [2,1,3,4]
# values1 = [10,5,15,4,6,14,16]
# tree = BinarySearchTree()
# for v in values1:
#   tree.append(v)
# dump_permutation_of_source_array(tree)