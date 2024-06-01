# !/usr/bin/env python
# coding: utf-8

'''
Description:
    Given a binary tree, return all root-to-leaf paths.

    For example, given the following binary tree:
           1
         /   \
        2     3
         \
          5
    All root-to-leaf paths are: ["1->2->5", "1->3"]

Tags: Tree, Depth-first Search
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param {TreeNode} root
    # @return {string[]}
    def binaryTreePaths(self, root):
        result, path = [], []
        self.binaryTreePathsRecu(self, node, path, result)
        return result

    def binaryTreePathsRecu(root, path, result):
        if node is None:
            return

        if node.left is node.right is None:
            ans = ''
            for n in path:
                ans += str(n.val) + '->'
            result.append(ans + str(node.val))

        if node.left:
            path.append(node)
            self.binaryTreePathsRecu(node.left, path, result)
            path.pop()

        if node.right:
            path.append(node)
            self.binaryTreePathsRecu(node.right, path, result)
            path.pop()
