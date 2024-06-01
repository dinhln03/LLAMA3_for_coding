from typing import Optional
from data_structures.singly_linked_list_node import SinglyLinkedListNode


def rotate_list(head: Optional[SinglyLinkedListNode], amount: int) -> Optional[SinglyLinkedListNode]:
    if not head:
        return None

    if not head.next:
        return head

    current = head
    number = 1

    while current.next:
        number += 1
        current = current.next
    current.next = head
    current = head

    for _ in range((number - amount) % number - 1):
        current = current.next

    new_head = current.next
    current.next = None
    return new_head
