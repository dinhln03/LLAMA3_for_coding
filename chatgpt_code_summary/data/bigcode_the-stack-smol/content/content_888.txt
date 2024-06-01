__source__ = 'https://leetcode.com/problems/delete-node-in-a-linked-list/description/'
# https://github.com/kamyu104/LeetCode/blob/master/Python/delete-node-in-a-linked-list.py
# Time:  O(1)
# Space: O(1)
#
# Description: Leetcode # 237. Delete Node in a Linked List
#
# Write a function to delete a node (except the tail) in a singly linked list,
# given only access to that node.
#
# Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are given the third node
# with value 3, the linked list should become 1 -> 2 -> 4 after calling your function.
#
# Companies
# Adobe Apple Microsoft
# Related Topics
# Linked List
# Similar Questions
# Remove Linked List Elements
#
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

import unittest
class Solution:
    # @param {ListNode} node
    # @return {void} Do not return anything, modify node in-place instead.
    def deleteNode(self, node):
        if node and node.next:
            node_to_delete = node.next
            node.val = node_to_delete.val
            node.next = node_to_delete.next
            del node_to_delete

class TestMethods(unittest.TestCase):
    def test_Local(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()

Java = '''
# Thought: https://leetcode.com/problems/delete-node-in-a-linked-list/solution/

Thought: We can't really delete the node, but we can kinda achieve the same effect
by instead removing the next node after copying its data into the node that we were asked to delete.

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */

# 0ms 100%
class Solution {
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }
}

# 0ms 100%
class Solution {
    public void deleteNode(ListNode node) {
        if (node == null || node.next == null) {
            return;
        }
        while (node.next.next != null) {
            node.val = node.next.val;
            node = node.next;
        }
        node.val = node.next.val;
        node.next = null;
    }
}
'''