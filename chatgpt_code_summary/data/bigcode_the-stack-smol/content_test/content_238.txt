from utils import TreeNode, binary_tree


class Solution:

	def __init__(self):
		self.index = 0 # 利用[中序遍历左边元素数量 = 左子树节点总数]可以省掉这个计数的字段

	def buildTree(self, preorder, inorder):
		"""
		:type preorder: List[int]
		:type inorder: List[int]
		:rtype: TreeNode
		"""
		if not preorder:
			return None

		def build_node(lo, hi):
			node = TreeNode(preorder[self.index])
			self.index += 1
			j = inorder.index(node.val, lo, hi) # 有些解法生成字典加快这步，但这会增大空间复杂度

			if self.index < len(preorder) and preorder[self.index] in inorder[lo:j]:
				node.left = build_node(lo, j)
			if self.index < len(preorder) and preorder[self.index] in inorder[j + 1:hi]:
				node.right = build_node(j + 1, hi)
			return node

		return build_node(0, len(preorder))


if __name__ == '__main__':
	x = Solution().buildTree([1, 2, 4, 6, 5, 7, 8, 3, 9], [4, 6, 2, 7, 5, 8, 1, 9, 3])
	x = Solution().buildTree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])
