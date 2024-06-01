# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        while head:
            if head.val == val:
                head = head.next
            else:
                break

        if not head:
            return head

        cur = head
        pre = cur
        while cur:
            if cur.val != val:
                pre = cur
                cur = cur.next
            else:
                cur = cur.next
                pre.next = cur
        return head
