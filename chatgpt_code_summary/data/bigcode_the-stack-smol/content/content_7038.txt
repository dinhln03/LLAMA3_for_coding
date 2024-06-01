def invert_binary_tree(node):
    if node:
        node.left, node.right = invert_binary_tree(node.right), invert_binary_tree(node.left)
        return node


class BinaryTreeNode(object):
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = None
        self.right = None

btn_root = BinaryTreeNode(10)
btn_1 = BinaryTreeNode(8)
btn_2 = BinaryTreeNode(9)
btn_root.left = btn_1
btn_root.right = btn_2
print(btn_root.left.value)
print(btn_root.right.value)
btn_root = invert_binary_tree(btn_root)
print(btn_root.left.value)
print(btn_root.right.value)
