import unittest
from gen_bst_seq import gen_bst_seq, Node

class Test_Case_Gen_Bst_Seq(unittest.TestCase):
    def test_gen_bst_seq(self):
        root = Node(2)
        root.left_child = Node(1)
        root.right_child = Node(3)
        root.right_child.left_child = Node(4)
        ans = gen_bst_seq(root)
        self.assertListEqual(ans, [[2, 1, 3, 4], [2, 3, 4, 1]])