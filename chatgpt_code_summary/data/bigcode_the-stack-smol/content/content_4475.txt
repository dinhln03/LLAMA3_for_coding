# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return None
        
        odd, even = ListNode(), ListNode()
        oddTail, evenTail = odd, even
        
        count = 0
        while head:
            if count % 2 == 0:
                evenTail.next = head
                evenTail = evenTail.next
            else:
                oddTail.next = head
                oddTail = oddTail.next
            head = head.next
            count += 1
        
        evenTail.next = odd.next
        oddTail.next = None
        
        return even.next
