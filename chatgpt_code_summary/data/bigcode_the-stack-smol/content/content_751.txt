# https://www.hackerrank.com/challenges/tree-height-of-a-binary-tree/problem


def height(root):
    """
    DFS

    v = Vertices
    e = Edges
    d = Depth

    Time complexity:   O(v + e)
    Space complexity:  O(d)
    """
    if root:
        return 1 + max(height(root.left), height(root.right))
    else:
        return -1
