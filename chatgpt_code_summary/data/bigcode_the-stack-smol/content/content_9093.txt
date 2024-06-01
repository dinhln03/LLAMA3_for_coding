from __future__ import annotations

def recursive_binary_search(l: list, target: int, low: int, high: int) -> int | None:
    mid = low + (high - low) // 2
    if mid < len(l):
        if l[mid] == target:
            return mid
        elif target > l[mid]:
            return recursive_binary_search(l, target, mid + 1, high)
        else:
            return recursive_binary_search(l, target, low, mid - 1)
    return None


if __name__ == '__main__':
    l = [3, 4, 8, 9, 14, 34, 41, 49, 58, 65, 69, 77, 81, 85, 88]
    print(recursive_binary_search(l, 3, 0, len(l)))
    print(recursive_binary_search(l, 41, 0, len(l)))
    print(recursive_binary_search(l, 88, 0, len(l)))
    print(recursive_binary_search(l, 89, 0, len(l)))