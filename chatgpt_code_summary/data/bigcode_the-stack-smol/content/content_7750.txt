def mergeSort(_list):
    n = len(_list)
    if n > 1:
        mid = n // 2    # int
        left = _list[:mid]
        right = _list[mid:]
        mergeSort(left)
        mergeSort(right)

        i = j = k = 0
        # 左右比較
        while i < len(left) and j < len(right):
            if left[i] < right[j]:  # left right compared
                _list[k] = left[i]
                i += 1
            else:
                _list[k] = right[j]
                j += 1
            k += 1
        # 看有沒有剩，直接塞滿
        while i < len(left):
            _list[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            _list[k] = right[j]
            j += 1
            k += 1

